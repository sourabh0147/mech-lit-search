import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET
import re
import logging
from typing import List, Dict, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Literature Search & Analytics", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Session State Initialization
# -----------------------------
def initialize_session_state():
    """Initialize session state variables"""
    if "papers" not in st.session_state:
        st.session_state.papers = pd.DataFrame()
    if "selected_papers" not in st.session_state:
        st.session_state.selected_papers = pd.DataFrame()
    if "semantic_key" not in st.session_state:
        st.session_state.semantic_key = ""
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

# -----------------------------
# Input Validation Functions
# -----------------------------
def validate_search_query(query: str) -> Optional[str]:
    """Validate and sanitize search query"""
    if not query or len(query.strip()) < 3:
        st.error("Please enter a search query with at least 3 characters")
        return None
    
    # Remove potentially problematic characters and sanitize
    sanitized = re.sub(r'[<>\"\'&]', '', query.strip())
    if len(sanitized) > 200:
        st.warning("Query truncated to 200 characters for optimal performance")
        sanitized = sanitized[:200]
    
    return sanitized

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate dataframe structure"""
    if df.empty:
        return False
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
        return False
    
    return True

# -----------------------------
# Enhanced Search Functions
# -----------------------------
def search_crossref(query: str, rows: int = 20) -> List[Dict]:
    """Search CrossRef database with enhanced error handling"""
    url = f"https://api.crossref.org/works"
    params = {
        'query': query,
        'rows': rows,
        'select': 'title,author,issued,DOI,abstract'
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        items = response.json().get("message", {}).get("items", [])
        results = []
        
        for item in items:
            # Extract title safely
            title = ""
            if "title" in item and item["title"]:
                title = item["title"][0] if isinstance(item["title"], list) else str(item["title"])
            
            # Extract authors safely
            authors = ""
            if "author" in item and item["author"]:
                author_names = []
                for author in item["author"]:
                    family = author.get("family", "")
                    given = author.get("given", "")
                    if family or given:
                        full_name = f"{given} {family}".strip()
                        author_names.append(full_name)
                authors = ", ".join(author_names)
            
            # Extract year safely
            year = None
            if "issued" in item and "date-parts" in item["issued"]:
                date_parts = item["issued"]["date-parts"]
                if date_parts and len(date_parts[0]) > 0:
                    year = date_parts[0][0]
            
            results.append({
                "title": title,
                "authors": authors,
                "year": year,
                "doi": item.get("DOI", ""),
                "abstract": item.get("abstract", ""),
                "source": "CrossRef",
            })
        
        return results
        
    except requests.exceptions.Timeout:
        st.error("CrossRef search timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        st.error("Connection error while searching CrossRef. Please check your internet connection.")
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error occurred while searching CrossRef: {e}")
    except Exception as e:
        st.error(f"Unexpected error during CrossRef search: {str(e)}")
        logger.error(f"CrossRef search error: {e}")
    
    return []

def search_arxiv(query: str, max_results: int = 20) -> List[Dict]:
    """Search arXiv database with enhanced error handling"""
    # Sanitize query for arXiv API
    clean_query = re.sub(r'[^\w\s\-\+\(\)]', '', query)
    url = f"http://export.arxiv.org/api/query"
    params = {
        'search_query': f'all:{clean_query}',
        'start': 0,
        'max_results': max_results
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        results = []
        
        for entry in root.findall('atom:entry', ns):
            # Extract title
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip() if title_elem is not None else ""
            
            # Extract authors
            author_elems = entry.findall('atom:author/atom:name', ns)
            authors = ", ".join([a.text for a in author_elems if a.text])
            
            # Extract year from published date
            published_elem = entry.find('atom:published', ns)
            year = None
            if published_elem is not None:
                try:
                    year = int(published_elem.text[:4])
                except (ValueError, TypeError):
                    year = None
            
            # Extract ID and abstract
            id_elem = entry.find('atom:id', ns)
            doi = id_elem.text if id_elem is not None else ""
            
            abstract_elem = entry.find('atom:summary', ns)
            abstract = abstract_elem.text.strip() if abstract_elem is not None else ""
            
            results.append({
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
                "abstract": abstract,
                "source": "arXiv",
            })
        
        return results
        
    except requests.exceptions.Timeout:
        st.error("arXiv search timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        st.error("Connection error while searching arXiv. Please check your internet connection.")
    except ET.ParseError:
        st.error("Error parsing arXiv response. Please try a different query.")
    except Exception as e:
        st.error(f"Unexpected error during arXiv search: {str(e)}")
        logger.error(f"arXiv search error: {e}")
    
    return []

def search_semantic_scholar(query: str, api_key: str = "", limit: int = 20) -> List[Dict]:
    """Search Semantic Scholar database"""
    if not api_key:
        return []
    
    headers = {'x-api-key': api_key} if api_key else {}
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': query, 
        'limit': limit, 
        'fields': 'title,authors,year,abstract,doi,publicationDate'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        papers = response.json().get('data', [])
        results = []
        
        for paper in papers:
            # Extract authors
            authors = ""
            if paper.get('authors'):
                author_names = [author.get('name', '') for author in paper['authors']]
                authors = ", ".join(filter(None, author_names))
            
            results.append({
                "title": paper.get('title', ''),
                "authors": authors,
                "year": paper.get('year'),
                "doi": paper.get('doi', ''),
                "abstract": paper.get('abstract', ''),
                "source": "Semantic Scholar"
            })
        
        return results
        
    except requests.exceptions.Timeout:
        st.error("Semantic Scholar search timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            st.error("Invalid Semantic Scholar API key. Please check your key.")
        else:
            st.error(f"Semantic Scholar API error: {e}")
    except Exception as e:
        st.error(f"Unexpected error during Semantic Scholar search: {str(e)}")
        logger.error(f"Semantic Scholar search error: {e}")
    
    return []

def merge_results(*args) -> pd.DataFrame:
    """Merge search results from multiple sources with deduplication"""
    all_results = [result for result_list in args for result in result_list]
    
    if not all_results:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    # Clean and deduplicate
    df['title'] = df['title'].fillna('').astype(str)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Deduplicate by title (case-insensitive)
    df['title_lower'] = df['title'].str.lower().str.strip()
    df = df.drop_duplicates(subset=['title_lower'], keep='first')
    df = df.drop(columns=['title_lower'])
    
    # Sort by year (most recent first)
    df = df.sort_values('year', ascending=False, na_last=True)
    
    return df.reset_index(drop=True)

# -----------------------------
# Enhanced Analytics Functions
# -----------------------------
def plot_keyword_frequency(df: pd.DataFrame) -> None:
    """Generate keyword frequency analysis with improved text processing"""
    if not validate_dataframe(df, ['abstract']):
        st.info("No data available for keyword analysis")
        return
    
    abstracts = df["abstract"].fillna("").astype(str)
    valid_abstracts = [abs_text for abs_text in abstracts if abs_text.strip()]
    
    if not valid_abstracts:
        st.info("No abstracts available for keyword analysis")
        return
    
    # Enhanced text processing
    text = " ".join(valid_abstracts)
    # Remove special characters and normalize
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    
    # Filter out stop words and short words
    filtered_words = [
        word for word in words 
        if word not in ENGLISH_STOP_WORDS 
        and len(word) > 2 
        and word.isalpha()
    ]
    
    if not filtered_words:
        st.info("No meaningful keywords found after filtering")
        return
    
    word_freq = pd.Series(filtered_words).value_counts().head(20)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    word_freq.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title("Top 20 Keywords in Abstracts", fontsize=14, fontweight='bold')
    ax.set_xlabel("Keywords", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # Prevent memory leaks
    
    # Create word cloud
    if len(filtered_words) > 10:  # Only create if enough words
        try:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(" ".join(filtered_words))
            
            fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            ax_wc.set_title("Word Cloud of Keywords", fontsize=14, fontweight='bold')
            st.pyplot(fig_wc)
            plt.close(fig_wc)
        except Exception as e:
            st.warning(f"Could not generate word cloud: {str(e)}")

def plot_publication_trend(df: pd.DataFrame) -> None:
    """Plot publication trends with enhanced data handling"""
    if not validate_dataframe(df, ['year']):
        st.info("No data available for publication trend analysis")
        return
    
    # Clean and filter years
    years = pd.to_numeric(df['year'], errors='coerce').dropna()
    current_year = pd.Timestamp.now().year
    valid_years = years[(years >= 1950) & (years <= current_year)]
    
    if valid_years.empty:
        st.info("No valid publication years found for trend analysis")
        return
    
    trend = valid_years.value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    trend.plot(kind='line', ax=ax, marker='o', linewidth=2, markersize=4, color='darkgreen')
    ax.set_title("Publication Trend Over Time", fontsize=14, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Publications", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Papers", len(valid_years))
    with col2:
        st.metric("Year Range", f"{int(valid_years.min())}-{int(valid_years.max())}")
    with col3:
        st.metric("Peak Year", f"{int(trend.idxmax())} ({trend.max()} papers)")

def plot_topic_clusters(df: pd.DataFrame) -> None:
    """Perform topic clustering with enhanced algorithm"""
    if not validate_dataframe(df, ['abstract', 'title']):
        st.info("No data available for clustering analysis")
        return
    
    valid_abstracts = df['abstract'].fillna("").astype(str)
    valid_data = valid_abstracts[valid_abstracts.str.strip() != ""]
    
    if len(valid_data) < 3:
        st.info("Insufficient data for clustering analysis (minimum 3 abstracts required)")
        return
    
    try:
        # Enhanced vectorization
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=500,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        X = vectorizer.fit_transform(valid_data)
        
        if X.shape[0] < 3:
            st.info("Insufficient unique content for clustering")
            return
        
        # Determine optimal number of clusters
        n_samples = X.shape[0]
        n_clusters = min(5, max(2, n_samples // 3))
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        labels = kmeans.fit_predict(X)
        
        # Create results dataframe
        cluster_df = pd.DataFrame({
            'cluster': labels,
            'title': df.loc[valid_data.index, 'title'].fillna("Untitled"),
            'abstract_preview': valid_data.str[:100] + "..."
        })
        
        st.subheader("Topic Clustering Results")
        
        # Display cluster distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            cluster_counts.plot(kind='bar', ax=ax, color='coral')
            ax.set_title("Papers per Cluster", fontsize=12, fontweight='bold')
            ax.set_xlabel("Cluster ID")
            ax.set_ylabel("Number of Papers")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.write("**Cluster Distribution:**")
            for cluster_id, count in cluster_counts.items():
                st.write(f"Cluster {cluster_id}: {count} papers")
        
        # Display papers by cluster
        st.subheader("Papers by Cluster")
        for cluster_id in sorted(cluster_counts.index):
            cluster_papers = cluster_df[cluster_df['cluster'] == cluster_id]
            with st.expander(f"Cluster {cluster_id} ({len(cluster_papers)} papers)"):
                st.dataframe(
                    cluster_papers[['title', 'abstract_preview']],
                    use_container_width=True
                )
        
    except Exception as e:
        st.error(f"Clustering analysis failed: {str(e)}")
        logger.error(f"Clustering error: {e}")

def plot_similarity_heatmap(df: pd.DataFrame) -> None:
    """Generate similarity heatmap with enhanced visualization"""
    if not validate_dataframe(df, ['abstract']):
        st.info("No data available for similarity analysis")
        return
    
    valid_abstracts = df['abstract'].fillna("").astype(str)
    valid_data = valid_abstracts[valid_abstracts.str.strip() != ""]
    
    if len(valid_data) < 2:
        st.info("Insufficient data for similarity analysis (minimum 2 abstracts required)")
        return
    
    if len(valid_data) > 20:
        st.warning("Large dataset detected. Showing similarity for first 20 papers only.")
        valid_data = valid_data.head(20)
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(valid_data)
        similarity_matrix = cosine_similarity(X)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_title("Document Similarity Heatmap", fontsize=14, fontweight='bold')
        ax.set_xlabel("Document Index")
        ax.set_ylabel("Document Index")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Display similarity statistics
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Similarity", f"{np.mean(upper_triangle):.3f}")
        with col2:
            st.metric("Max Similarity", f"{np.max(upper_triangle):.3f}")
        with col3:
            st.metric("Min Similarity", f"{np.min(upper_triangle):.3f}")
        
    except Exception as e:
        st.error(f"Similarity analysis failed: {str(e)}")
        logger.error(f"Similarity analysis error: {e}")

# -----------------------------
# Enhanced Export Functions
# -----------------------------
def export_selected_papers() -> None:
    """Enhanced export functionality with multiple formats"""
    if st.session_state.selected_papers.empty:
        st.info("No papers selected for export")
        return
    
    st.subheader("Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv_data = st.session_state.selected_papers.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"selected_papers_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # JSON Export
        json_data = st.session_state.selected_papers.to_json(
            orient='records', 
            indent=2, 
            date_format='iso'
        )
        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name=f"selected_papers_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        # BibTeX Export (simplified)
        def create_bibtex():
            bibtex_entries = []
            for _, row in st.session_state.selected_papers.iterrows():
                # Create a simple BibTeX entry
                title = row.get('title', 'Untitled').replace('{', '').replace('}', '')
                authors = row.get('authors', 'Unknown')
                year = row.get('year', 'Unknown')
                doi = row.get('doi', '')
                
                entry_id = f"{authors.split(',')[0].replace(' ', '')}_{year}".lower()
                entry = f"""@article{{{entry_id},
    title = {{{title}}},
    author = {{{authors}}},
    year = {{{year}}},
    doi = {{{doi}}}
}}

"""
                bibtex_entries.append(entry)
            
            return ''.join(bibtex_entries)
        
        bibtex_data = create_bibtex()
        st.download_button(
            label="📚 Download BibTeX",
            data=bibtex_data,
            file_name=f"selected_papers_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.bib",
            mime="text/plain",
            use_container_width=True
        )

# -----------------------------
# Main Application Interface
# -----------------------------
def main():
    """Main application function"""
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.title("🔬 Configuration")
        
        # API Keys section
        st.subheader("API Keys (Optional)")
        st.session_state.semantic_key = st.text_input(
            "Semantic Scholar API Key",
            value=st.session_state.semantic_key,
            type="password",
            help="Enter your Semantic Scholar API key for additional search results"
        )
        
        # Search settings
        st.subheader("Search Settings")
        max_results_per_source = st.slider("Max results per source", 5, 50, 20)
        
        # Display search history
        if st.session_state.search_history:
            st.subheader("Recent Searches")
            for query in st.session_state.search_history[-5:]:
                if st.button(f"🔄 {query[:30]}...", key=f"hist_{query}"):
                    st.session_state.current_query = query
    
    # Main navigation
    page = st.sidebar.radio(
        "📋 Navigation", 
        ["🔍 Search", "📑 Selected Papers", "📊 Analytics"],
        format_func=lambda x: x.split(" ", 1)[1]
    )
    
    # Page routing
    if "🔍 Search" in page:
        render_search_page(max_results_per_source)
    elif "📑 Selected Papers" in page:
        render_selected_papers_page()
    elif "📊 Analytics" in page:
        render_analytics_page()

def render_search_page(max_results_per_source: int):
    """Render the search page"""
    st.title("🔍 Literature Search")
    st.markdown("---")
    
    # Search interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., machine learning in healthcare, quantum computing applications",
            help="Enter keywords related to your research topic"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Advanced search options
    with st.expander("⚙️ Advanced Search Options"):
        col1, col2 = st.columns(2)
        with col1:
            search_crossref = st.checkbox("Search CrossRef", value=True)
            search_arxiv = st.checkbox("Search arXiv", value=True)
        with col2:
            search_semantic = st.checkbox(
                "Search Semantic Scholar", 
                value=bool(st.session_state.semantic_key),
                disabled=not bool(st.session_state.semantic_key)
            )
            if not st.session_state.semantic_key:
                st.caption("⚠️ Requires API key")
    
    # Perform search
    if search_button and query:
        validated_query = validate_search_query(query)
        if validated_query:
            # Add to search history
            if validated_query not in st.session_state.search_history:
                st.session_state.search_history.append(validated_query)
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            total_sources = sum([search_crossref, search_arxiv, search_semantic])
            current_source = 0
            
            # Search CrossRef
            if search_crossref:
                status_text.text("Searching CrossRef...")
                progress_bar.progress(current_source / total_sources)
                crossref_results = search_crossref(validated_query, max_results_per_source)
                results.append(crossref_results)
                current_source += 1
                st.success(f"Found {len(crossref_results)} results from CrossRef")
            
            # Search arXiv
            if search_arxiv:
                status_text.text("Searching arXiv...")
                progress_bar.progress(current_source / total_sources)
                arxiv_results = search_arxiv(validated_query, max_results_per_source)
                results.append(arxiv_results)
                current_source += 1
                st.success(f"Found {len(arxiv_results)} results from arXiv")
            
            # Search Semantic Scholar
            if search_semantic and st.session_state.semantic_key:
                status_text.text("Searching Semantic Scholar...")
                progress_bar.progress(current_source / total_sources)
                semantic_results = search_semantic_scholar(
                    validated_query, 
                    st.session_state.semantic_key, 
                    max_results_per_source
                )
                results.append(semantic_results)
                current_source += 1
                st.success(f"Found {len(semantic_results)} results from Semantic Scholar")
            
            # Merge and display results
            status_text.text("Processing results...")
            progress_bar.progress(1.0)
            
            st.session_state.papers = merge_results(*results)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            total_results = len(st.session_state.papers)
            if total_results > 0:
                st.success(f"🎉 Found {total_results} unique papers total")
            else:
                st.warning("No results found. Try adjusting your search terms.")
    
    # Display search results
    if not st.session_state.papers.empty:
        st.markdown("---")
        st.subheader(f"📚 Search Results ({len(st.session_state.papers)} papers)")
        
        # Results summary
        col1, col2, col3 = st.columns(3)
        with col1:
            source_counts = st.session_state.papers['source'].value_counts()
            st.metric("Sources", len(source_counts))
        with col2:
            year_range = st.session_state.papers['year'].dropna()
            if not year_range.empty:
                st.metric("Year Range", f"{int(year_range.min())}-{int(year_range.max())}")
        with col3:
            selected_count = len(st.session_state.selected_papers)
            st.metric("Selected Papers", selected_count)
        
        # Display papers
        for i, row in st.session_state.papers.iterrows():
            with st.expander(
                f"📄 {row['title'][:80]}{'...' if len(str(row['title'])) > 80 else ''} ({row['source']})"
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Authors:** {row['authors'] if row['authors'] else 'Not available'}")
                    st.write(f"**Year:** {row['year'] if pd.notna(row['year']) else 'Not available'}")
                    if row['doi']:
                        st.write(f"**DOI:** {row['doi']}")
                    
                    abstract = row['abstract'] if row['abstract'] else "No abstract available"
                    st.write(f"**Abstract:** {abstract}")
                
                with col2:
                    if st.button(f"➕ Add to Selection", key=f"add_{i}"):
                        # Check if already selected
                        if not st.session_state.selected_papers.empty:
                            existing_titles = st.session_state.selected_papers['title'].str.lower()
                            if row['title'].lower() not in existing_titles.values:
                                st.session_state.selected_papers = pd.concat([
                                    st.session_state.selected_papers, 
                                    pd.DataFrame([row])
                                ], ignore_index=True)
                                st.success("Added to selection!")
                            else:
                                st.warning("Already in selection")
                        else:
                            st.session_state.selected_papers = pd.DataFrame([row])
                            st.success("Added to selection!")

def render_selected_papers_page():
    """Render the selected papers page"""
    st.title("📑 Selected Papers")
    st.markdown("---")
    
    if st.session_state.selected_papers.empty:
        st.info("📭 No papers selected yet. Go to the Search page to find and select papers.")
        return
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Papers", len(st.session_state.selected_papers))
    with col2:
        sources = st.session_state.selected_papers['source'].nunique()
        st.metric("Unique Sources", sources)
    with col3:
        years = st.session_state.selected_papers['year'].dropna()
        if not years.empty:
            st.metric("Latest Year", int(years.max()))
    with col4:
        abstracts_available = st.session_state.selected_papers['abstract'].notna().sum()
        st.metric("With Abstracts", abstracts_available)
    
    # Management buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.selected_papers = pd.DataFrame()
            st.success("All papers cleared!")
            st.rerun()
    
    with col2:
        show_details = st.checkbox("Show Details", value=False)
    
    with col3:
        sort_option = st.selectbox("Sort by", ["Year (Desc)", "Year (Asc)", "Title", "Source"])
    
    # Sort papers
    df_display = st.session_state.selected_papers.copy()
    if sort_option == "Year (Desc)":
        df_display = df_display.sort_values('year', ascending=False, na_last=True)
    elif sort_option == "Year (Asc)":
        df_display = df_display.sort_values('year', ascending=True, na_last=False)
    elif sort_option == "Title":
        df_display = df_display.sort_values('title', ascending=True, na_last=True)
    elif sort_option == "Source":
        df_display = df_display.sort_values('source', ascending=True, na_last=True)
    
    # Display papers
    st.markdown("---")
    
    if show_details:
        # Detailed view
        for i, (idx, row) in enumerate(df_display.iterrows()):
            with st.expander(f"📄 {row['title']}", expanded=False):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"**Authors:** {row['authors'] if row['authors'] else 'Not available'}")
                    st.write(f"**Year:** {row['year'] if pd.notna(row['year']) else 'Not available'}")
                    st.write(f"**Source:** {row['source']}")
                    if row['doi']:
                        st.write(f"**DOI:** {row['doi']}")
                    if row['abstract']:
                        st.write(f"**Abstract:** {row['abstract']}")
                
                with col2:
                    if st.button(f"❌ Remove", key=f"remove_{idx}"):
                        st.session_state.selected_papers = st.session_state.selected_papers.drop(idx).reset_index(drop=True)
                        st.success("Paper removed!")
                        st.rerun()
    else:
        # Table view
        display_columns = ['title', 'authors', 'year', 'source']
        st.dataframe(
            df_display[display_columns],
            use_container_width=True,
            height=400
        )
    
    # Export section
    st.markdown("---")
    export_selected_papers()

def render_analytics_page():
    """Render the analytics page"""
    st.title("📊 Analytics Dashboard")
    st.markdown("---")
    
    if st.session_state.selected_papers.empty:
        st.info("📭 No papers selected for analysis. Go to the Search page to find and select papers.")
        return
    
    df = st.session_state.selected_papers
    
    # Analytics summary
    st.subheader("📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Papers", len(df))
    with col2:
        abstracts_count = df['abstract'].notna().sum()
        st.metric("Papers with Abstracts", f"{abstracts_count}/{len(df)}")
    with col3:
        unique_authors = set()
        for authors in df['authors'].dropna():
            unique_authors.update([a.strip() for a in str(authors).split(',')])
        st.metric("Unique Authors", len(unique_authors))
    with col4:
        year_span = df['year'].dropna()
        if not year_span.empty:
            span = int(year_span.max()) - int(year_span.min()) + 1
            st.metric("Year Span", f"{span} years")
    
    # Analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔤 Keywords", "📅 Timeline", "🎯 Clusters", "🔗 Similarity"])
    
    with tab1:
        st.subheader("Keyword Frequency Analysis")
        plot_keyword_frequency(df)
    
    with tab2:
        st.subheader("Publication Timeline")
        plot_publication_trend(df)
    
    with tab3:
        st.subheader("Topic Clustering")
        plot_topic_clusters(df)
    
    with tab4:
        st.subheader("Document Similarity Analysis")
        plot_similarity_heatmap(df)

# -----------------------------
# Application Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
