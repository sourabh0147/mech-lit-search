import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

st.set_page_config(page_title="Literature Search & Analytics", layout="wide")

# -------------------------
# Retry-enabled CrossRef Search
# -------------------------
def search_crossref(query, limit=50, retries=3):
    url = f"https://api.crossref.org/works?query={query}&rows={limit}"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("items", [])
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                st.error("CrossRef search timed out after multiple attempts.")
                return []
        except Exception as e:
            st.error(f"CrossRef error: {e}")
            return []

# -------------------------
# ArXiv Search
# -------------------------
def search_arxiv(query, limit=50):
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={limit}"
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.text  # You can parse XML if needed
    except Exception as e:
        st.error(f"ArXiv error: {e}")
        return []

# -------------------------
# Sidebar Navigation
# -------------------------
menu = st.sidebar.radio("Navigation", ["Search", "Selected Papers", "Analytics"])

if 'selected_papers' not in st.session_state:
    st.session_state.selected_papers = []

# -------------------------
# Search Page
# -------------------------
if menu == "Search":
    st.title("🔍 Literature Search")
    query = st.text_input("Enter your search query")
    crossref_enabled = st.checkbox("Search CrossRef", value=True)
    arxiv_enabled = st.checkbox("Search ArXiv", value=True)
    max_results = st.slider("Max results per source", 10, 100, 50, 10)

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a valid search query.")
        else:
            results = []
            if crossref_enabled:
                st.info("Fetching results from CrossRef...")
                crossref_results = search_crossref(query, max_results)
                results.extend(crossref_results)
            if arxiv_enabled:
                st.info("Fetching results from ArXiv...")
                arxiv_results = search_arxiv(query, max_results)
                results.extend(arxiv_results if isinstance(arxiv_results, list) else [])

            if results:
                for i, paper in enumerate(results):
                    title = paper.get("title", ["Untitled"])[0] if isinstance(paper, dict) else str(paper)
                    with st.container():
                        st.markdown(f"**{i+1}. {title}**")
                        if isinstance(paper, dict):
                            st.caption(paper.get("DOI", "No DOI"))
                        if st.button(f"Add {i+1}", key=f"add_{i}"):
                            st.session_state.selected_papers.append(title)
            else:
                st.warning("No results found.")

# -------------------------
# Selected Papers
# -------------------------
if menu == "Selected Papers":
    st.title("📄 Selected Papers")
    if st.session_state.selected_papers:
        st.write(st.session_state.selected_papers)
        if st.button("Clear Selection"):
            st.session_state.selected_papers = []
    else:
        st.info("No papers selected yet.")

# -------------------------
# Analytics
# -------------------------
if menu == "Analytics":
    st.title("📊 Analytics")
    if st.session_state.selected_papers:
        text_data = " ".join(st.session_state.selected_papers)

        # Keyword frequency
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(st.session_state.selected_papers)
        feature_names = vectorizer.get_feature_names_out()
        freqs = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        freq_df = pd.DataFrame({'keyword': feature_names, 'frequency': freqs}).sort_values(by='frequency', ascending=False).head(20)

        st.subheader("Top Keywords")
        st.bar_chart(freq_df.set_index('keyword'))

        # Wordcloud
        wc = WordCloud(width=800, height=400, background_color='white').generate(text_data)
        st.subheader("Wordcloud")
        st.image(wc.to_array(), use_container_width=True)

        # Topic Clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix)
        st.subheader("Topic Clusters")
        st.write(pd.DataFrame({"Paper": st.session_state.selected_papers, "Cluster": clusters}))

        # Similarity Heatmap
        sim_matrix = cosine_similarity(tfidf_matrix)
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.matshow(sim_matrix, cmap='coolwarm')
        plt.title("Similarity Heatmap", pad=20)
        plt.colorbar(cax)
        st.pyplot(fig)
    else:
        st.info("Please select papers first.")
