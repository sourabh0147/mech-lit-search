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
import altair as alt

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

            if results:
                for i, paper in enumerate(results):
                    title = paper.get("title", ["Untitled"])[0]
                    authors = ", ".join([auth.get("given", "") + " " + auth.get("family", "") for auth in paper.get("author", [])]) if paper.get("author") else "Unknown Authors"
                    year = paper.get("issued", {}).get("date-parts", [["Unknown"]])[0][0]
                    doi = paper.get("DOI", "")
                    with st.expander(f"{i+1}. {title}"):
                        st.write(f"**Authors:** {authors}")
                        st.write(f"**Year:** {year}")
                        if doi:
                            st.markdown(f"[Open DOI Link](https://doi.org/{doi})")
                        if st.button(f"Add to Selection", key=f"add_{i}"):
                            st.session_state.selected_papers.append({
                                "title": title,
                                "authors": authors,
                                "year": year,
                                "doi": doi
                            })
            else:
                st.warning("No results found.")

# -------------------------
# Selected Papers
# -------------------------
if menu == "Selected Papers":
    st.title("📄 Selected Papers")
    if st.session_state.selected_papers:
        df = pd.DataFrame(st.session_state.selected_papers)
        st.dataframe(df)
        st.download_button("Download as CSV", df.to_csv(index=False), "selected_papers.csv")
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
        df = pd.DataFrame(st.session_state.selected_papers)
        text_data = " ".join(df['title'].tolist())

        # Keyword frequency
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['title'])
        feature_names = vectorizer.get_feature_names_out()
        freqs = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        freq_df = pd.DataFrame({'keyword': feature_names, 'frequency': freqs}).sort_values(by='frequency', ascending=False).head(20)

        st.subheader("Top Keywords")
        chart = alt.Chart(freq_df).mark_bar().encode(
            x=alt.X('frequency', title='Frequency'),
            y=alt.Y('keyword', sort='-x', title='Keyword')
        ).properties(width=600, height=400)
        st.altair_chart(chart, use_container_width=True)

        # Wordcloud
        wc = WordCloud(width=800, height=400, background_color='white').generate(text_data)
        st.subheader("Wordcloud")
        st.image(wc.to_array(), use_container_width=True)

        # Topic Clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix)
        df['Cluster'] = clusters
        st.subheader("Topic Clusters")
        st.dataframe(df[['title', 'Cluster']])

        # Similarity Heatmap
        st.subheader("Similarity Heatmap")
        sim_matrix = cosine_similarity(tfidf_matrix)
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.matshow(sim_matrix, cmap='coolwarm')
        plt.title("Similarity Heatmap", pad=20)
        plt.colorbar(cax)
        st.pyplot(fig)
    else:
        st.info("Please select papers first.")
