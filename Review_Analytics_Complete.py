"""
Minimal Working Literature Search App
Version: Emergency Fix - Works with minimal dependencies
This version checks for missing dependencies and provides fallbacks
"""

import streamlit as st
import pandas as pd
import sys
import subprocess
import importlib
from typing import List, Dict, Any
import json
from datetime import datetime

# ================== Dependency Checker ==================

def check_and_install_dependencies():
    """Check for required dependencies and attempt to install missing ones."""
    
    required_packages = {
        'requests': 'requests',
        'feedparser': 'feedparser',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    optional_packages = {
        'plotly': 'plotly',
        'arxiv': 'arxiv',
        'scholarly': 'scholarly',
        'matplotlib': 'matplotlib'
    }
    
    missing_required = []
    missing_optional = []
    available_packages = {}
    
    # Check required packages
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            available_packages[module_name] = True
        except ImportError:
            missing_required.append(package_name)
            available_packages[module_name] = False
    
    # Check optional packages
    for module_name, package_name in optional_packages.items():
        try:
            importlib.import_module(module_name)
            available_packages[module_name] = True
        except ImportError:
            missing_optional.append(package_name)
            available_packages[module_name] = False
    
    return available_packages, missing_required, missing_optional

# Check dependencies at startup
AVAILABLE_PACKAGES, MISSING_REQUIRED, MISSING_OPTIONAL = check_and_install_dependencies()

# ================== Safe Imports ==================

# Required imports with error handling
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    st.error("❌ Critical Error: 'requests' module not found. Please install it: `pip install requests`")

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    st.warning("⚠️ 'feedparser' module not found. ArXiv search will be limited.")

# Optional imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False

# ================== Configuration ==================

st.set_page_config(
    page_title="Literature Search - Emergency Mode",
    page_icon="🔍",
    layout="wide"
)

# ================== Paper Class ==================

class Paper:
    """Simple paper representation."""
    def __init__(self, title="", authors=None, year=0, abstract="", 
                 journal="", doi="", url="", source="", citations=0):
        self.title = title
        self.authors = authors or []
        self.year = year
        self.abstract = abstract
        self.journal = journal
        self.doi = doi
        self.url = url
        self.source = source
        self.citations = citations
    
    def to_dict(self):
        return {
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'abstract': self.abstract,
            'journal': self.journal,
            'doi': self.doi,
            'url': self.url,
            'source': self.source,
            'citations': self.citations
        }
    
    def to_bibtex(self):
        """Convert to BibTeX format."""
        authors_str = " and ".join(self.authors) if self.authors else "Unknown"
        key = f"paper{self.year}"
        
        return f"""@article{{{key},
    title = {{{self.title}}},
    author = {{{authors_str}}},
    year = {{{self.year}}},
    journal = {{{self.journal}}},
    doi = {{{self.doi}}},
    url = {{{self.url}}}
}}"""

# ================== API Functions ==================

def search_semantic_scholar(query: str, limit: int = 10) -> List[Paper]:
    """Search Semantic Scholar API."""
    papers = []
    
    if not REQUESTS_AVAILABLE:
        st.error("Cannot search: requests module not available")
        return papers
    
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,authors,year,abstract,venue,citationCount,url'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            for item in data.get('data', []):
                paper = Paper(
                    title=item.get('title', ''),
                    authors=[a.get('name', '') for a in item.get('authors', [])],
                    year=item.get('year', 0),
                    abstract=item.get('abstract', '')[:500] if item.get('abstract') else '',
                    journal=item.get('venue', ''),
                    url=item.get('url', ''),
                    source='Semantic Scholar',
                    citations=item.get('citationCount', 0)
                )
                papers.append(paper)
        else:
            st.error(f"Semantic Scholar API error: {response.status_code}")
            
    except Exception as e:
        st.error(f"Semantic Scholar search failed: {str(e)}")
    
    return papers

def search_crossref(query: str, limit: int = 10) -> List[Paper]:
    """Search CrossRef API."""
    papers = []
    
    if not REQUESTS_AVAILABLE:
        return papers
    
    try:
        url = "https://api.crossref.org/works"
        params = {
            'query': query,
            'rows': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            for item in data.get('message', {}).get('items', []):
                # Extract authors
                authors = []
                for author in item.get('author', []):
                    name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                    if name:
                        authors.append(name)
                
                # Extract year
                year = 0
                if 'published-print' in item:
                    date_parts = item['published-print'].get('date-parts', [[]])
                    if date_parts and date_parts[0]:
                        year = date_parts[0][0]
                
                paper = Paper(
                    title=' '.join(item.get('title', [''])),
                    authors=authors,
                    year=year,
                    abstract=item.get('abstract', '')[:500] if item.get('abstract') else '',
                    journal=' '.join(item.get('container-title', [''])),
                    doi=item.get('DOI', ''),
                    url=item.get('URL', ''),
                    source='CrossRef',
                    citations=item.get('is-referenced-by-count', 0)
                )
                papers.append(paper)
        else:
            st.error(f"CrossRef API error: {response.status_code}")
            
    except Exception as e:
        st.error(f"CrossRef search failed: {str(e)}")
    
    return papers

def search_arxiv_simple(query: str, limit: int = 10) -> List[Paper]:
    """Search arXiv using HTTP API (no special library needed)."""
    papers = []
    
    if not REQUESTS_AVAILABLE:
        return papers
    
    if not FEEDPARSER_AVAILABLE:
        st.warning("feedparser not available - ArXiv search disabled")
        return papers
    
    try:
        url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': query,
            'start': 0,
            'max_results': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            feed = feedparser.parse(response.text)
            
            for entry in feed.entries:
                # Extract authors
                authors = [author.get('name', '') for author in entry.get('authors', [])]
                
                # Extract year from published date
                year = 0
                if 'published' in entry:
                    year = int(entry.published[:4])
                
                paper = Paper(
                    title=entry.get('title', '').replace('\n', ' '),
                    authors=authors,
                    year=year,
                    abstract=entry.get('summary', '')[:500],
                    journal='arXiv',
                    url=entry.get('link', ''),
                    source='arXiv'
                )
                papers.append(paper)
        else:
            st.error(f"arXiv API error: {response.status_code}")
            
    except Exception as e:
        st.error(f"arXiv search failed: {str(e)}")
    
    return papers

# ================== Main Application ==================

def main():
    st.title("📚 Academic Literature Search")
    st.caption("Emergency Mode - Working with Available APIs")
    
    # Show dependency status
    with st.sidebar:
        st.header("System Status")
        
        if MISSING_REQUIRED:
            st.error("Missing Required Packages:")
            for pkg in MISSING_REQUIRED:
                st.write(f"❌ {pkg}")
            st.code(f"pip install {' '.join(MISSING_REQUIRED)}")
        
        st.subheader("API Status")
        st.write(f"{'✅' if REQUESTS_AVAILABLE else '❌'} Basic APIs")
        st.write(f"{'✅' if FEEDPARSER_AVAILABLE else '❌'} ArXiv")
        st.write(f"{'✅' if PLOTLY_AVAILABLE else '❌'} Advanced Charts")
        st.write(f"{'✅' if SCHOLARLY_AVAILABLE else '❌'} Google Scholar")
        
        if st.button("🔧 Install Missing Packages"):
            st.info("Run this command in your terminal:")
            st.code("pip install -r requirements.txt")
    
    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    
    # Search interface
    st.header("Search Papers")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter search query",
            placeholder="e.g., 'deep learning mechanical fault detection'"
        )
    
    with col2:
        st.write("")
        st.write("")
        search_button = st.button("🔍 Search", type="primary")
    
    # Search options
    with st.expander("Search Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Select Databases:**")
            use_semantic = st.checkbox("Semantic Scholar", value=True, disabled=not REQUESTS_AVAILABLE)
            use_crossref = st.checkbox("CrossRef", value=True, disabled=not REQUESTS_AVAILABLE)
            use_arxiv = st.checkbox("arXiv", value=True, disabled=not FEEDPARSER_AVAILABLE)
        
        with col2:
            results_per_db = st.number_input("Results per database", 5, 50, 10)
    
    # Execute search
    if search_button and query:
        all_papers = []
        
        with st.spinner("Searching..."):
            # Search each database
            if use_semantic:
                papers = search_semantic_scholar(query, results_per_db)
                all_papers.extend(papers)
                st.success(f"Found {len(papers)} papers from Semantic Scholar")
            
            if use_crossref:
                papers = search_crossref(query, results_per_db)
                all_papers.extend(papers)
                st.success(f"Found {len(papers)} papers from CrossRef")
            
            if use_arxiv:
                papers = search_arxiv_simple(query, results_per_db)
                all_papers.extend(papers)
                st.success(f"Found {len(papers)} papers from arXiv")
        
        st.session_state.search_results = all_papers
        st.success(f"Total: {len(all_papers)} papers found")
    
    # Display results
    if st.session_state.search_results:
        st.header(f"Search Results ({len(st.session_state.search_results)} papers)")
        
        for i, paper in enumerate(st.session_state.search_results, 1):
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"**{i}. {paper.title}**")
                    if paper.authors:
                        st.caption(f"Authors: {', '.join(paper.authors[:3])}")
                    st.caption(f"Year: {paper.year} | Source: {paper.source} | Citations: {paper.citations}")
                    
                    if paper.url:
                        st.markdown(f"[🔗 View Paper]({paper.url})")
                    
                    with st.expander("Abstract"):
                        st.write(paper.abstract or "No abstract available")
                
                with col2:
                    if st.button("Export BibTeX", key=f"bib_{i}"):
                        st.code(paper.to_bibtex(), language="bibtex")
                
                st.divider()
        
        # Export all results
        if st.button("📥 Export All Results (JSON)"):
            data = [p.to_dict() for p in st.session_state.search_results]
            json_str = json.dumps(data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    if not REQUESTS_AVAILABLE:
        st.error("""
        ## ❌ Critical Error: Missing Dependencies
        
        The application cannot run without the required packages.
        
        ### Please run this command in your terminal:
        ```bash
        pip install requests feedparser pandas numpy
        ```
        
        ### Or install all dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        
        After installation, refresh this page.
        """)
    else:
        main()
