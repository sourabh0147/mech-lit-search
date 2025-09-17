# mech_lit_search_app.py
import re
import json
import requests
import pandas as pd
import streamlit as st
from pathlib import Path

CONFIG_PATH = Path("config.json")  # relative path so it works in cloud too
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

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
# Main Search Function
# ---------------------------
def search_semantic_scholar(query, api_key=None):
    params = {
        "query": query,
        "limit": 20,
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
        else:
            st.warning(f"Semantic Scholar returned {r.status_code}")
            return pd.DataFrame()
    except requests.RequestException as e:
        st.error(f"Error contacting Semantic Scholar: {e}")
        return pd.DataFrame()

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.set_page_config(page_title="Mechanical Engineering Literature Search", layout="centered")
    st.title("🔍 Mechanical Engineering Literature Search")
    st.caption("Search and explore research papers — runs on any browser!")

    config = safe_load_json(CONFIG_PATH)
    with st.expander("⚙️ Advanced Settings", expanded=False):
        enhanced_mode = st.checkbox("Enable Enhanced Search (API Keys Required)", value=config.get("enhanced_mode", False))
        api_key = None
        if enhanced_mode:
            st.markdown("**Get your free API key here:** [Semantic Scholar API](https://www.semanticscholar.org/product/api)")
            api_key = st.text_input("🔑 Semantic Scholar API Key", value=config.get("semantic_scholar_api_key", ""), type="password")
            if st.button("💾 Save Credentials"):
                config["semantic_scholar_api_key"] = api_key
                config["enhanced_mode"] = True
                safe_save_json(CONFIG_PATH, config)
                st.success("Credentials saved!")

    query = st.text_input("Enter your search query", placeholder="e.g. tribology of magnesium alloys")
    if st.button("🔍 Search"):
        if query.strip():
            with st.spinner("Searching papers..."):
                results = search_semantic_scholar(clean_query(query), api_key if config.get("enhanced_mode") else None)
            if results.empty:
                st.error("No results found.")
            else:
                st.success(f"Found {len(results)} papers")
                st.dataframe(results, use_container_width=True)
        else:
            st.warning("Please enter a search term.")

if __name__ == "__main__":
    main()
