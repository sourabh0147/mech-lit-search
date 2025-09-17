# mech_lit_search_app_optimized.py
import re
import json
import requests
import pandas as pd
import streamlit as st
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

CONFIG_PATH = Path("config.json")
CACHE_PATH = Path("cache.json")

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_URL = "https://api.crossref.org/works"
DOAJ_URL = "https://doaj.org/api/v2/search/articles/"
ARXIV_URL = "http://export.arxiv.org/api/query"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

MAX_RESULTS_PER_SOURCE = 300  # Configurable max results per source

# ---------------------------
# Utility Functions
# ---------------------------
def safe_load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except:
            return {}
    return {}

def safe_save_json(path: Path, data: dict):
    try:
        path.write_text(json.dumps(data, indent=2))
    except:
        pass

def clean_query(query: str) -> str:
    query = re.sub(r"[^\w\s\-+\"']", " ", query)
    return " ".join(query.split())

# ---------------------------
# Caching
# ---------------------------
def get_cached_results(query):
    cache = safe_load_json(CACHE_PATH)
    return pd.DataFrame(cache.get(query, []))

def save_cache(query, df):
    cache = safe_load_json(CACHE_PATH)
    cache[query] = df.to_dict(orient="records")
    safe_save_json(CACHE_PATH, cache)

# ---------------------------
# Search Functions
# ---------------------------
def search_semantic_scholar(query, api_key=None):
    results = []
    limit = 100
    offset = 0
    while offset < MAX_RESULTS_PER_SOURCE:
        params = {
            "query": query,
            "limit": min(limit, MAX_RESULTS_PER_SOURCE - offset),
            "offset": offset,
            "fields": "paperId,title,authors,year,citationCount,externalIds,url"
        }
        headers = {"User-Agent": "MechEngSearch/2.0"}
        if api_key:
            headers["x-api-key"] = api_key
        try:
            r = requests.get(SEMANTIC_SCHOLAR_URL, params=params, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if not data:
                    break
                for p in data:
                    results.append({
                        "Source": "Semantic Scholar",
                        "Title": p.get("title"),
                        "Authors": ", ".join([a["name"] for a in p.get("authors", [])]),
                        "Year": p.get("year"),
                        "Citations": p.get("citationCount"),
                        "DOI": (p.get("externalIds") or {}).get("DOI"),
                        "URL": p.get("url")
                    })
                offset += len(data)
            elif r.status_code == 429:
                st.warning("Semantic Scholar API rate limit reached. Partial results returned.")
                break
            else:
                st.warning(f"Semantic Scholar returned {r.status_code}")
                break
        except Exception as e:
            st.warning(f"Semantic Scholar error: {e}")
            break
    return pd.DataFrame(results)

def search_crossref(query):
    results = []
    offset = 0
    rows_per_request = 100
    while offset < MAX_RESULTS_PER_SOURCE:
        params = {"query": query, "rows": min(rows_per_request, MAX_RESULTS_PER_SOURCE - offset), "offset": offset}
        try:
            r = requests.get(CROSSREF_URL, params=params, timeout=30)
            if r.status_code == 200:
                items = r.json()["message"]["items"]
                if not items:
                    break
                for i in items:
                    results.append({
                        "Source": "Crossref",
                        "Title": i.get("title", [""])[0],
                        "Authors": ", ".join([f"{a.get('given','')} {a.get('family','')}" for a in i.get("author", [])]) if "author" in i else "",
                        "Year": i.get("issued", {}).get("date-parts", [[None]])[0][0],
                        "Citations": None,
                        "DOI": i.get("DOI"),
                        "URL": i.get("URL")
                    })
                offset += len(items)
            else:
                st.warning(f"Crossref returned {r.status_code}")
                break
        except Exception as e:
            st.warning(f"Crossref error: {e}")
            break
    return pd.DataFrame(results)

def search_doaj(query):
    results = []
    page = 1
    page_size = 100
    while len(results) < MAX_RESULTS_PER_SOURCE:
        params = {"q": query, "pageSize": page_size, "page": page}
        try:
            r = requests.get(DOAJ_URL, params=params, timeout=20)
            if r.status_code == 200:
                items = r.json().get("results", [])
                if not items:
                    break
                for i in items:
                    bib = i.get("bibjson", {})
                    results.append({
                        "Source": "DOAJ",
                        "Title": bib.get("title"),
                        "Authors": ", ".join([a.get("name","") for a in bib.get("author", [])]),
                        "Year": bib.get("year"),
                        "Citations": None,
                        "DOI": bib.get("identifier", [{}])[0].get("id") if bib.get("identifier") else None,
                        "URL": bib.get("link", [{}])[0].get("url") if bib.get("link") else None
                    })
                page += 1
            else:
                st.warning(f"DOAJ returned {r.status_code}")
                break
        except Exception as e:
            st.warning(f"DOAJ error: {e}")
            break
    return pd.DataFrame(results[:MAX_RESULTS_PER_SOURCE])

def search_arxiv(query):
    results = []
    batch_size = 100
    start = 0
    while len(results) < MAX_RESULTS_PER_SOURCE:
        params = {"search_query": f"all:{query}", "start": start, "max_results": batch_size}
        try:
            r = requests.get(ARXIV_URL, params=params, timeout=20)
            if r.status_code != 200:
                st.warning(f"arXiv returned {r.status_code}")
                break
            root = ET.fromstring(r.text)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            if not entries:
                break
            for e in entries:
                title = e.find("{http://www.w3.org/2005/Atom}title").text
                authors = ", ".join([a.find("{http://www.w3.org/2005/Atom}name").text for a in e.findall("{http://www.w3.org/2005/Atom}author")])
                url = e.find("{http://www.w3.org/2005/Atom}id").text
                year = e.find("{http://www.w3.org/2005/Atom}published").text[:4]
                results.append({"Source": "arXiv", "Title": title, "Authors": authors, "Year": year, "Citations": None, "DOI": None, "URL": url})
            start += batch_size
        except Exception as e:
            st.warning(f"arXiv error: {e}")
            break
    return pd.DataFrame(results[:MAX_RESULTS_PER_SOURCE])

def search_pubmed(query):
    results = []
    retmax = 100
    retstart = 0
    while len(results) < MAX_RESULTS_PER_SOURCE:
        try:
            params_search = {"db": "pubmed", "term": query, "retmax": min(retmax, MAX_RESULTS_PER_SOURCE - len(results)), "retstart": retstart, "retmode": "json"}
            r = requests.get(PUBMED_SEARCH_URL, params=params_search, timeout=20)
            if r.status_code != 200:
                st.warning(f"PubMed search returned {r.status_code}")
                break
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                break
            params_summary = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
            r_summary = requests.get(PUBMED_SUMMARY_URL, params=params_summary, timeout=20)
            data = r_summary.json().get("result", {})
            for pid in ids:
                item = data.get(pid)
                if item:
                    authors = ", ".join([a["name"] for a in item.get("authors", [])])
                    results.append({
                        "Source": "PubMed",
                        "Title": item.get("title"),
                        "Authors": authors,
                        "Year": item.get("pubdate", "")[:4],
                        "Citations": None,
                        "DOI": item.get("elocationid"),
                        "URL": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
                    })
            retstart += retmax
        except Exception as e:
            st.warning(f"PubMed error: {e}")
            break
    return pd.DataFrame(results[:MAX_RESULTS_PER_SOURCE])

# ---------------------------
# Combined Search with ThreadPoolExecutor
# ---------------------------
def combined_search(query, api_key=None):
    cached = get_cached_results(query)
    if not cached.empty:
        st.info("Showing cached results")
        return cached

    results_dfs = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(func, query): name
            for func, name in [
                (search_semantic_scholar, "Semantic Scholar"),
                (search_crossref, "Crossref"),
                (search_doaj, "DOAJ"),
                (search_arxiv, "arXiv"),
                (search_pubmed, "PubMed")
            ]
        }
        for future in as_completed(futures):
            df = future.result()
            if not df.empty:
                results_dfs.append(df)
                st.success(f"Fetched {len(df)} results from {futures[future]}")
                st.dataframe(df[["Source","Title","Authors","Year"]].head(5))  # Incremental display

    combined = pd.concat(results_dfs, ignore_index=True) if results_dfs else pd.DataFrame()
    if not combined.empty:
        save_cache(query, combined)
    else:
        st.error("No results found from any source. Please try again later.")
    return combined

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(page_title="Mechanical Engineering Literature Search", page_icon="🔎", layout="wide")

    st.markdown("""
    <style>
    h1,h2,h3 { font-family: 'Segoe UI', sans-serif; }
    .stButton button { background-color:#2563EB;color:white;border-radius:10px;padding:0.5em 1em;font-size:1.1em; }
    .stButton button:hover { background-color:#1E40AF; }
    .paper-card { background:white; padding:1em; margin-bottom:1em; border-radius:12px; box-shadow:0px 2px 8px rgba(0,0,0,0.1);}
    .paper-title { font-weight:600; font-size:1.2em; margin-bottom:0.2em; }
    .paper-meta { color:#475569; font-size:0.9em; }
    .link-btn { font-size:0.85em; background-color:#e2e8f0; border-radius:6px; padding:3px 8px; margin-right:6px; text-decoration:none;}
    .link-btn:hover { background-color:#cbd5e1; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔎 Mechanical Engineering Literature Search")
    st.caption("Search multiple sources including Semantic Scholar, Crossref, DOAJ, arXiv, and PubMed.")

    config = safe_load_json(CONFIG_PATH)
    with st.expander("⚙️ Advanced Settings", expanded=False):
        enhanced_mode = st.checkbox("Enable Enhanced Search (Semantic Scholar API Key Required)", value=config.get("enhanced_mode", False))
        api_key = None
        if enhanced_mode:
            st.markdown("**Get your free API key:** [Semantic Scholar API](https://www.semanticscholar.org/product/api)")
            api_key = st.text_input("🔑 Semantic Scholar API Key", value=config.get("semantic_scholar_api_key",""), type="password")
            if st.button("💾 Save Credentials"):
                config["semantic_scholar_api_key"] = api_key
                config["enhanced_mode"] = True
                safe_save_json(CONFIG_PATH, config)
                st.success("Credentials saved!")

    query = st.text_input("Enter your search query", placeholder="e.g. tribology of magnesium alloys")
    if st.button("🔍 Search"):
        if query.strip():
            with st.spinner("Fetching papers from all sources..."):
                results = combined_search(clean_query(query), api_key if config.get("enhanced_mode") else None)

            if not results.empty:
                st.subheader(f"📚 Total {len(results)} results")
                for _, row in results.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="paper-card">
                            <div class="paper-title">{row['Title']}</div>
                            <div class="paper-meta">
                                <b>Source:</b> {row['Source']}<br>
                                <b>Authors:</b> {row['Authors']} <br>
                                <b>Year:</b> {row['Year']} | <b>Citations:</b> {row['Citations'] if row['Citations'] else 'N/A'}
                            </div>
                            <div style="margin-top:0.5em;">
                                <a class="link-btn" href="{row['URL']}" target="_blank">View Paper</a>
                                {"<a class='link-btn' href='https://doi.org/" + row['DOI'] + "' target='_blank'>DOI</a>" if row['DOI'] else ""}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a search term.")

if __name__ == "__main__":
    main()
