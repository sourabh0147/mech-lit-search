# mech_lit_search_app.py
import re
import json
import requests
import pandas as pd
import streamlit as st
from pathlib import Path
import time

CONFIG_PATH = Path("config.json")
CACHE_PATH = Path("cache.json")

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_URL = "https://api.crossref.org/works"
DOAJ_URL = "https://doaj.org/api/v2/search/articles/"
ARXIV_URL = "http://export.arxiv.org/api/query"
PUBMED_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# ---------------------------
# Utility Functions
# ---------------------------
def safe_load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}

def safe_save_json(path: Path, data: dict):
    try:
        path.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

def clean_query(query: str) -> str:
    query = re.sub(r"[^\w\s\-+\"']", " ", query)
    return " ".join(query.split())

def top_n_terms(query: str, n=3):
    words = re.findall(r"\b[A-Za-z]{4,}\b", query)
    return " ".join(words[:n]) if words else query

# ---------------------------
# Cache Functions
# ---------------------------
def get_cached_results(query):
    cache = safe_load_json(CACHE_PATH)
    return pd.DataFrame(cache.get(query, []))

def save_cache(query, df):
    cache = safe_load_json(CACHE_PATH)
    cache[query] = df.to_dict(orient="records")
    safe_save_json(CACHE_PATH, cache)

# ---------------------------
# Semantic Scholar
# ---------------------------
def search_semantic_scholar(query, api_key=None):
    params = {
        "query": query,
        "limit": 10,
        "fields": "paperId,title,authors,year,citationCount,externalIds,url"
    }
    headers = {"User-Agent": "MechEngSearch/2.0"}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        r = requests.get(SEMANTIC_SCHOLAR_URL, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", [])
            return pd.DataFrame([
                {
                    "Source": "Semantic Scholar",
                    "Title": p.get("title"),
                    "Authors": ", ".join([a["name"] for a in p.get("authors", [])]),
                    "Year": p.get("year"),
                    "Citations": p.get("citationCount"),
                    "DOI": (p.get("externalIds") or {}).get("DOI"),
                    "URL": p.get("url")
                }
                for p in data
            ])
        elif r.status_code == 400:
            return search_semantic_scholar(top_n_terms(query), api_key)
        elif r.status_code == 429:
            st.warning("Semantic Scholar API rate limit reached.")
            return pd.DataFrame()
        else:
            st.warning(f"Semantic Scholar returned {r.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Semantic Scholar error: {e}")
        return pd.DataFrame()

# ---------------------------
# Crossref
# ---------------------------
def search_crossref(query, retries=2):
    params = {"query": query, "rows": 10}
    for attempt in range(retries + 1):
        try:
            r = requests.get(CROSSREF_URL, params=params, timeout=30)
            if r.status_code == 200:
                items = r.json()["message"]["items"]
                return pd.DataFrame([
                    {
                        "Source": "Crossref",
                        "Title": i.get("title", [""])[0],
                        "Authors": ", ".join([f"{a.get('given','')} {a.get('family','')}" for a in i.get("author", [])]) if "author" in i else "",
                        "Year": i.get("issued", {}).get("date-parts", [[None]])[0][0],
                        "Citations": None,
                        "DOI": i.get("DOI"),
                        "URL": i.get("URL")
                    }
                    for i in items
                ])
            else:
                st.warning(f"Crossref returned {r.status_code}")
                return pd.DataFrame()
        except requests.exceptions.ReadTimeout:
            if attempt < retries:
                time.sleep(2)
            else:
                st.warning("Crossref API read timeout.")
                return pd.DataFrame()
        except Exception as e:
            st.warning(f"Crossref error: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# ---------------------------
# DOAJ
# ---------------------------
def search_doaj(query):
    try:
        params = {"q": query, "pageSize": 10}
        r = requests.get(DOAJ_URL, params=params, timeout=20)
        if r.status_code == 200:
            items = r.json().get("results", [])
            return pd.DataFrame([
                {
                    "Source": "DOAJ",
                    "Title": i.get("bibjson", {}).get("title"),
                    "Authors": ", ".join([a.get("name","") for a in i.get("bibjson", {}).get("author", [])]),
                    "Year": i.get("bibjson", {}).get("year"),
                    "Citations": None,
                    "DOI": i.get("bibjson", {}).get("identifier", [{}])[0].get("id") if i.get("bibjson", {}).get("identifier") else None,
                    "URL": i.get("bibjson", {}).get("link", [{}])[0].get("url") if i.get("bibjson", {}).get("link") else None
                } for i in items
            ])
        else:
            st.warning(f"DOAJ returned {r.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"DOAJ error: {e}")
        return pd.DataFrame()

# ---------------------------
# arXiv
# ---------------------------
def search_arxiv(query):
    try:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 10
        }
        r = requests.get(ARXIV_URL, params=params, timeout=20)
        if r.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(r.text)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            results = []
            for e in entries:
                title = e.find("{http://www.w3.org/2005/Atom}title").text
                authors = ", ".join([a.find("{http://www.w3.org/2005/Atom}name").text for a in e.findall("{http://www.w3.org/2005/Atom}author")])
                url = e.find("{http://www.w3.org/2005/Atom}id").text
                year = e.find("{http://www.w3.org/2005/Atom}published").text[:4]
                results.append({"Source": "arXiv", "Title": title, "Authors": authors, "Year": year, "Citations": None, "DOI": None, "URL": url})
            return pd.DataFrame(results)
        else:
            st.warning(f"arXiv returned {r.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"arXiv error: {e}")
        return pd.DataFrame()

# ---------------------------
# PubMed
# ---------------------------
def search_pubmed(query):
    try:
        # Step 1: ESearch to get IDs
        params_search = {
            "db": "pubmed",
            "term": query,
            "retmax": 10,
            "retmode": "json"
        }
        r = requests.get(PUBMED_URL, params=params_search, timeout=20)
        if r.status_code != 200:
            st.warning(f"PubMed search returned {r.status_code}")
            return pd.DataFrame()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return pd.DataFrame()
        # Step 2: ESummary to get details
        params_summary = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json"
        }
        r = requests.get(PUBMED_SUMMARY_URL, params=params_summary, timeout=20)
        if r.status_code != 200:
            st.warning(f"PubMed summary returned {r.status_code}")
            return pd.DataFrame()
        data = r.json().get("result", {})
        results = []
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
        return pd.DataFrame(results)
    except Exception as e:
        st.warning(f"PubMed error: {e}")
        return pd.DataFrame()

# ---------------------------
# Combined Search
# ---------------------------
def combined_search(query, api_key=None):
    # Check cache
    cached = get_cached_results(query)
    if not cached.empty:
        st.info("Showing cached results")
        return cached

    results_list = [
        search_semantic_scholar(query, api_key),
        search_crossref(query),
        search_doaj(query),
        search_arxiv(query),
        search_pubmed(query)
    ]
    combined = pd.concat([df for df in results_list if not df.empty], ignore_index=True)
    if combined.empty:
        st.error("No results found from any source. Please try again later.")
    else:
        save_cache(query, combined)
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
                st.subheader(f"📚 Showing {len(results)} results")
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
                            <div style="margin-top: 0.5em;">
                                <a class="link-btn" href="{row['URL']}" target="_blank">View Paper</a>
                                {"<a class='link-btn' href='https://doi.org/" + row['DOI'] + "' target='_blank'>DOI</a>" if row['DOI'] else ""}
                            </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a search term.")

if __name__ == "__main__":
    main()
