# mech_lit_search_app_clean.py
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

MAX_RESULTS_PER_SOURCE = 300
THREADS = 5

# ---------------------------
# Utilities
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
    df = pd.DataFrame(cache.get(query, []))
    return df if not df.empty else None

def save_cache(query, df):
    if df.empty:
        return
    cache = safe_load_json(CACHE_PATH)
    cache[query] = df.to_dict(orient="records")
    safe_save_json(CACHE_PATH, cache)

# ---------------------------
# Search functions
# ---------------------------
def search_semantic_scholar(query, api_key=None):
    results, offset = [], 0
    while offset < MAX_RESULTS_PER_SOURCE:
        params = {
            "query": query,
            "limit": min(100, MAX_RESULTS_PER_SOURCE - offset),
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
                if not data: break
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
                st.warning("Semantic Scholar rate limit reached. Partial results returned.")
                break
            else:
                st.warning(f"Semantic Scholar returned {r.status_code}")
                break
        except Exception as e:
            st.warning(f"Semantic Scholar error: {e}")
            break
    return pd.DataFrame(results)

def search_crossref(query):
    results, offset = [], 0
    while offset < MAX_RESULTS_PER_SOURCE:
        params = {"query": query, "rows": min(100, MAX_RESULTS_PER_SOURCE - offset), "offset": offset}
        try:
            r = requests.get(CROSSREF_URL, params=params, timeout=30)
            if r.status_code != 200: break
            items = r.json()["message"]["items"]
            if not items: break
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
        except Exception as e:
            st.warning(f"Crossref error: {e}")
            break
    return pd.DataFrame(results)

def search_doaj(query):
    results, page = [], 1
    while len(results) < MAX_RESULTS_PER_SOURCE:
        params = {"q": query, "pageSize": 100, "page": page}
        try:
            r = requests.get(DOAJ_URL, params=params, timeout=20)
            if r.status_code != 200: break
            items = r.json().get("results", [])
            if not items: break
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
        except Exception as e:
            st.warning(f"DOAJ error: {e}")
            break
    return pd.DataFrame(results[:MAX_RESULTS_PER_SOURCE])

def search_arxiv(query):
    results, start = [], 0
    while len(results) < MAX_RESULTS_PER_SOURCE:
        params = {"search_query": f"all:{query}", "start": start, "max_results": 100}
        try:
            r = requests.get(ARXIV_URL, params=params, timeout=20)
            if r.status_code != 200: break
            root = ET.fromstring(r.text)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            if not entries: break
            for e in entries:
                title = e.find("{http://www.w3.org/2005/Atom}title").text
                authors = ", ".join([a.find("{http://www.w3.org/2005/Atom}name").text for a in e.findall("{http://www.w3.org/2005/Atom}author")])
                url = e.find("{http://www.w3.org/2005/Atom}id").text
                year = e.find("{http://www.w3.org/2005/Atom}published").text[:4]
                results.append({"Source": "arXiv", "Title": title, "Authors": authors, "Year": year, "Citations": None, "DOI": None, "URL": url})
            start += 100
        except Exception as e:
            st.warning(f"arXiv error: {e}")
            break
    return pd.DataFrame(results[:MAX_RESULTS_PER_SOURCE])

def search_pubmed(query):
    results, retstart = [], 0
    while len(results) < MAX_RESULTS_PER_SOURCE:
        retmax = min(100, MAX_RESULTS_PER_SOURCE - len(results))
        try:
            r = requests.get(PUBMED_SEARCH_URL, params={"db":"pubmed","term":query,"retmax":retmax,"retstart":retstart,"retmode":"json"}, timeout=20)
            if r.status_code != 200: break
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if not ids: break
            r_sum = requests.get(PUBMED_SUMMARY_URL, params={"db":"pubmed","id":",".join(ids),"retmode":"json"}, timeout=20)
            data = r_sum.json().get("result", {})
            for pid in ids:
                item = data.get(pid)
                if item:
                    authors = ", ".join([a["name"] for a in item.get("authors", [])])
                    results.append({
                        "Source": "PubMed",
                        "Title": item.get("title"),
                        "Authors": authors,
                        "Year": item.get("pubdate","")[:4],
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
# Display results cleanly
# ---------------------------
def display_results(df):
    for _, row in df.iterrows():
        doi_link = f'<a style="font-size:0.85em;background:#e2e8f0;border-radius:6px;padding:3px 8px;margin-right:6px;text-decoration:none;" href="https://doi.org/{row["DOI"]}" target="_blank">DOI</a>' if row["DOI"] else ""
        st.markdown(f'''
        <div style="background:white;padding:1em;margin-bottom:1em;border-radius:12px;box-shadow:0px 2px 8px rgba(0,0,0,0.1);">
            <div style="font-weight:600;font-size:1.2em;margin-bottom:0.2em;">{row["Title"]}</div>
            <div style="color:#475569;font-size:0.9em;">
                <b>Source:</b> {row["Source"]}<br>
                <b>Authors:</b> {row["Authors"]}<br>
                <b>Year:</b> {row["Year"]} | <b>Citations:</b> {row["Citations"]}
            </div>
            <span style="margin-top:0.5em;">
                <a style="font-size:0.85em;background:#e2e8f0;border-radius:6px;padding:3px 8px;margin-right:6px;text-decoration:none;" href="{row['URL']}" target="_blank">View Paper</a>
                {doi_link}
            </span>
        </div>
        ''', unsafe_allow_html=True)

# ---------------------------
# Combined multi-source search
# ---------------------------
def combined_search(query, api_key=None, selected_sources=None):
    if selected_sources is None:
        selected_sources = ["Semantic Scholar","Crossref","DOAJ","arXiv","PubMed"]

    results = []
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        future_to_source = {}
        if "Semantic Scholar" in selected_sources:
            future_to_source[executor.submit(search_semantic_scholar, query, api_key)] = "Semantic Scholar"
        if "Crossref" in selected_sources:
            future_to_source[executor.submit(search_crossref, query)] = "Crossref"
        if "DOAJ" in selected_sources:
            future_to_source[executor.submit(search_doaj, query)] = "DOAJ"
        if "arXiv" in selected_sources:
            future_to_source[executor.submit(search_arxiv, query)] = "arXiv"
        if "PubMed" in selected_sources:
            future_to_source[executor.submit(search_pubmed, query)] = "PubMed"

        for future in as_completed(future_to_source):
            df = future.result()
            if not df.empty:
                results.append(df)

    if results:
        combined_df = pd.concat(results, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(page_title="Mechanical Engineering Literature Search", page_icon="🔎", layout="wide")
    st.title("🔎 Mechanical Engineering Literature Search")
    st.caption("Multi-source paper search: Semantic Scholar, Crossref, DOAJ, arXiv, PubMed")

    config = safe_load_json(CONFIG_PATH)

    with st.expander("⚙️ Advanced Settings", expanded=False):
        enhanced_mode = st.checkbox("Enable Enhanced Search (API Keys Required)", value=config.get("enhanced_mode", False))
        api_key = None
        if enhanced_mode:
            st.markdown("**Get your free API key:** [Semantic Scholar API](https://www.semanticscholar.org/product/api)")
            api_key = st.text_input("🔑 Semantic Scholar API Key", value=config.get("semantic_scholar_api_key", ""), type="password")
            if st.button("💾 Save Credentials"):
                config["semantic_scholar_api_key"] = api_key
                config["enhanced_mode"] = True
                safe_save_json(CONFIG_PATH, config)
                st.success("Credentials saved!")

        st.markdown("### Filters")
        year_range = st.slider("Publication Year Range", 1900, 2030, (2000, 2025))
        min_citations = st.number_input("Minimum Citations (0 = ignore)", min_value=0, value=0)
        selected_sources = st.multiselect("Select Sources", ["Semantic Scholar","Crossref","DOAJ","arXiv","PubMed"], default=["Semantic Scholar","Crossref","DOAJ","arXiv","PubMed"])

    query = st.text_input("Enter your search query", placeholder="e.g. tribology of magnesium alloys")

    if st.button("🔍 Search"):
        if query.strip():
            cached = get_cached_results(query)
            if cached is not None:
                st.info("Showing cached results")
                display_results(cached)
            else:
                with st.spinner("Fetching papers..."):
                    results = combined_search(
                        clean_query(query),
                        api_key if enhanced_mode else None,
                        selected_sources=selected_sources
                    )
                    if not results.empty:
                        # Apply filters
                        results = results[results["Year"].apply(lambda x: x is not None and year_range[0] <= int(x) <= year_range[1])]
                        if min_citations > 0:
                            results = results[results["Citations"].fillna(0).astype(int) >= min_citations]

                        display_results(results)
                        save_cache(query, results)
                    else:
                        st.error("No results found from the selected sources.")
        else:
            st.warning("Please enter a search term.")

if __name__ == "__main__":
    main()
