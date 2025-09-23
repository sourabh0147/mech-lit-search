import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Literature Search & Analytics", layout="wide")

# -----------------------------
# Session State Initialization
# -----------------------------
if "papers" not in st.session_state:
    st.session_state.papers = pd.DataFrame()
if "selected_papers" not in st.session_state:
    st.session_state.selected_papers = pd.DataFrame()
if "semantic_key" not in st.session_state:
    st.session_state.semantic_key = ""

# -----------------------------
# Helper Functions
# -----------------------------
def search_crossref(query, rows=20):
    url = f"https://api.crossref.org/works?query={query}&rows={rows}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            items = r.json()["message"]["items"]
            return [
                {
                    "title": i.get("title", [""])[0],
                    "authors": ", ".join([a.get("family", "") for a in i.get("author", [])]) if "author" in i else "",
                    "year": i.get("issued", {}).get("date-parts", [[None]])[0][0],
                    "doi": i.get("DOI", ""),
                    "abstract": i.get("abstract", ""),
                    "source": "CrossRef",
                }
                for i in items
            ]
    except:
        return []
    return []

def search_arxiv(query, max_results=20):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(r.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            results = []
            for entry in root.findall('atom:entry', ns):
                results.append({
                    "title": entry.find('atom:title', ns).text.strip(),
                    "authors": ", ".join([a.text for a in entry.findall('atom:author/atom:name', ns)]),
                    "year": entry.find('atom:published', ns).text[:4],
                    "doi": entry.find('atom:id', ns).text,
                    "abstract": entry.find('atom:summary', ns).text.strip(),
                    "source": "arXiv",
                })
            return results
    except:
        return []
    return []

def merge_results(*args):
    df = pd.DataFrame([x for lst in args for x in lst])
    if df.empty:
        return df
    return df.drop_duplicates(subset=["title"])  # deduplicate by title

# -----------------------------
# Analytics Functions
# -----------------------------
def plot_keyword_frequency(df):
    text = " ".join(df["abstract"].fillna(""))
    words = pd.Series(text.split()).value_counts().head(20)
    fig, ax = plt.subplots(figsize=(8,4))
    words.plot(kind='bar', ax=ax)
    ax.set_title("Top Keywords")
    st.pyplot(fig)
    wc = WordCloud(width=600, height=300, background_color='white').generate(text)
    st.image(wc.to_array(), caption="WordCloud of Keywords")

def plot_publication_trend(df):
    trend = df['year'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6,3))
    trend.plot(ax=ax)
    ax.set_title("Publication Trend")
    ax.set_xlabel("Year")
    st.pyplot(fig)

def plot_topic_clusters(df):
    if df['abstract'].dropna().empty:
        st.info("No abstracts available for clustering")
        return
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(df['abstract'].fillna(""))
    kmeans = KMeans(n_clusters=min(5, len(df)), random_state=42)
    labels = kmeans.fit_predict(X)
    df['cluster'] = labels
    st.write("Cluster counts:", df['cluster'].value_counts())

def plot_similarity_heatmap(df):
    if df['abstract'].dropna().empty:
        st.info("No abstracts available for similarity heatmap")
        return
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['abstract'].fillna(""))
    sim_matrix = cosine_similarity(X)
    fig, ax = plt.subplots(figsize=(5,5))
    cax = ax.imshow(sim_matrix, cmap='viridis')
    ax.set_title("Similarity Heatmap")
    fig.colorbar(cax)
    st.pyplot(fig)

# -----------------------------
# Sidebar & Navigation
# -----------------------------
page = st.sidebar.radio("Navigation", ["Search", "Selected Papers", "Analytics"])

with st.sidebar:
    st.subheader("Optional API Keys")
    st.session_state.semantic_key = st.text_input("Semantic Scholar API Key", value=st.session_state.semantic_key)

# -----------------------------
# Main Pages
# -----------------------------
if page == "Search":
    st.title("Literature Search")
    query = st.text_input("Enter search query")
    if st.button("Search") and query:
        with st.spinner("Searching papers..."):
            crossref_results = search_crossref(query)
            arxiv_results = search_arxiv(query)
            st.session_state.papers = merge_results(crossref_results, arxiv_results)
        st.success(f"Found {len(st.session_state.papers)} papers")

    if not st.session_state.papers.empty:
        for i, row in st.session_state.papers.iterrows():
            with st.expander(f"{row['title']} ({row['source']})"):
                st.write(f"**Authors:** {row['authors']}")
                st.write(f"**Year:** {row['year']}")
                st.write(row['abstract'] if row['abstract'] else "No abstract available")
                if st.button(f"Add {i}", key=f"add_{i}"):
                    st.session_state.selected_papers = pd.concat([st.session_state.selected_papers, pd.DataFrame([row])], ignore_index=True)

elif page == "Selected Papers":
    st.title("Selected Papers")
    if st.session_state.selected_papers.empty:
        st.info("No papers selected")
    else:
        st.dataframe(st.session_state.selected_papers[['title','authors','year','source']])
        st.download_button("Download CSV", st.session_state.selected_papers.to_csv(index=False), "selected_papers.csv")

elif page == "Analytics":
    st.title("Analytics")
    df = st.session_state.selected_papers
    if df.empty:
        st.info("Please select papers first")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Keyword Frequency", "Publication Trend", "Topic Clusters", "Similarity Heatmap"])
        with tab1:
            plot_keyword_frequency(df)
        with tab2:
            plot_publication_trend(df)
        with tab3:
            plot_topic_clusters(df)
        with tab4:
            plot_similarity_heatmap(df)
