import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import altair as alt
from itertools import cycle

st.set_page_config(page_title="Literature Search & Analytics", layout="wide")

# -------------------------
# API Search Functions
# -------------------------
def search_crossref(query, limit=50, retries=3):
    url = f"https://api.crossref.org/works?query={query}&rows={limit}"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("items", [])
        except:
            time.sleep(2)
    return []

def search_springer(query, limit=50, retries=3):
    api_key = st.secrets.get("SPRINGER_API_KEY")
    if not api_key:
        return []
    url = f"http://api.springernature.com/metadata/json?q={query}&p={limit}&api_key={api_key}"
    for _ in range(retries):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            return data.get("records", [])
        except:
            time.sleep(2)
    return []

def search_elsevier(query, limit=50, retries=3):
    api_key = st.secrets.get("ELSEVIER_API_KEY")
    if not api_key:
        return []
    headers = {"X-ELS-APIKey": api_key}
    url = f"https://api.elsevier.com/content/search/scopus?query={query}&count={limit}"
    for _ in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
            return data.get("search-results", {}).get("entry", [])
        except:
            time.sleep(2)
    return []

# -------------------------
# Session State
# -------------------------
if 'selected_papers' not in st.session_state:
    st.session_state.selected_papers = []

# -------------------------
# Tabs UI
# -------------------------
tabs = st.tabs(["🔍 Search", "📄 Selected Papers", "📊 Analytics"])
search_tab, selected_tab, analytics_tab = tabs

# -------------------------
# SEARCH TAB
# -------------------------
with search_tab:
    st.title("Literature Search")
    query = st.text_input("Enter your search query")
    sources = st.multiselect("Select sources", ["CrossRef", "Springer", "Elsevier"], default=["CrossRef"])
    max_results = st.slider("Max results per source", 10, 100, 50, 10)

    if st.button("Search"):
        if not query.strip():
            st.warning("Enter a valid query")
        else:
            results = []
            if "CrossRef" in sources:
                results.extend(search_crossref(query, max_results))
            if "Springer" in sources:
                results.extend(search_springer(query, max_results))
            if "Elsevier" in sources:
                results.extend(search_elsevier(query, max_results))

            if results:
                colors = cycle(["#f9f9f9", "#e6f7ff", "#fff7e6"])
                for i, paper in enumerate(results):
                    title = paper.get("title", ["Untitled"])[0] if "title" in paper else paper.get("dc:title", "Untitled")
                    authors = ", ".join([auth.get("given", "") + " " + auth.get("family", "") for auth in paper.get("author", [])]) if paper.get("author") else paper.get("dc:creator", "Unknown")
                    year = paper.get("issued", {}).get("date-parts", [["Unknown"]])[0][0] if "issued" in paper else paper.get("prism:coverDate", "Unknown")[:4]
                    doi = paper.get("DOI", "") or paper.get("prism:doi", "")
                    journal = paper.get("container-title", [""])[0] if "container-title" in paper else paper.get("prism:publicationName", "")

                    with st.container():
                        st.markdown(f"""
                        <div style='background-color:{next(colors)};padding:10px;border-radius:8px;margin-bottom:8px'>
                        <h4>{i+1}. {title} ({year})</h4>
                        <p><b>Authors:</b> {authors}<br>
                        <b>Journal:</b> {journal}<br>
                        {"<b>DOI:</b> <a href='https://doi.org/"+doi+"' target='_blank'>Link</a>" if doi else ""}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"Add to Selection", key=f"add_{i}"):
                            st.session_state.selected_papers.append({
                                "title": title, "authors": authors, "year": year, "doi": doi, "journal": journal
                            })
            else:
                st.warning("No results found.")

# -------------------------
# SELECTED PAPERS TAB
# -------------------------
with selected_tab:
    st.title("Selected Papers")
    if st.session_state.selected_papers:
        df = pd.DataFrame(st.session_state.selected_papers)
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "selected_papers.csv")
        if st.button("Clear Selection"):
            st.session_state.selected_papers = []
    else:
        st.info("No papers selected yet.")

# -------------------------
# ANALYTICS TAB
# -------------------------
with analytics_tab:
    st.title("Analytics Dashboard")
    if st.session_state.selected_papers:
        df = pd.DataFrame(st.session_state.selected_papers)
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        text_data = " ".join(df['title'].tolist())

        # Two-column layout
        col1, col2 = st.columns([2, 1])

        # -------------------------
        # Keyword Frequency
        # -------------------------
        with col1:
            st.subheader("Top Keywords")
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(df['title'])
            feature_names = vectorizer.get_feature_names_out()
            freqs = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            freq_df = pd.DataFrame({'keyword': feature_names, 'frequency': freqs}).sort_values(by='frequency', ascending=False).head(20)
            chart = alt.Chart(freq_df).mark_bar().encode(
                x=alt.X('frequency', title='Frequency'),
                y=alt.Y('keyword', sort='-x', title='Keyword')
            ).properties(width=600, height=400)
            st.altair_chart(chart, use_container_width=True)

        # -------------------------
        # Publication Trend
        # -------------------------
        with col2:
            st.subheader("Publication Trend")
            trend_df = df.groupby('year').size().reset_index(name='count')
            trend_chart = alt.Chart(trend_df).mark_line(point=True).encode(
                x='year',
                y='count',
                tooltip=['year', 'count']
            ).interactive()
            st.altair_chart(trend_chart, use_container_width=True)

        # -------------------------
        # Wordcloud
        # -------------------------
        st.subheader("Wordcloud")
        wc = WordCloud(width=800, height=400, background_color='white').generate(text_data)
        st.image(wc.to_array(), use_container_width=True)

        # -------------------------
        # Topic Clustering
        # -------------------------
        st.subheader("Topic Clusters")
        k = min(5, len(df))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix)
        df['Cluster'] = clusters
        cluster_color_map = {i: color for i, color in enumerate(cycle(["#FFDDC1", "#C1FFD7", "#C1D4FF", "#FFF1C1", "#E0C1FF"]))}
        st.dataframe(df.style.apply(lambda row: [f'background-color: {cluster_color_map[row.Cluster]}' for _ in row], axis=1))

        # -------------------------
        # Similarity Heatmap
        # -------------------------
        st.subheader("Similarity Heatmap")
        sim_matrix = cosine_similarity(tfidf_matrix)
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.matshow(sim_matrix, cmap='coolwarm')
        plt.title("Similarity Heatmap", pad=20)
        plt.colorbar(cax)
        st.pyplot(fig)
    else:
        st.info("Please select papers first.")
