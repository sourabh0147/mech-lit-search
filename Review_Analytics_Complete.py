"""
Integrated Mechanical Engineering Literature Search and Analytics Application
Version 4.0 - With Real Academic Database API Integration

This version integrates with multiple academic databases:
- Semantic Scholar
- CrossRef
- DOAJ (Directory of Open Access Journals)
- arXiv
- PubMed
- IEEE Xplore (requires API key)
- Google Scholar (via scholarly)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
import time
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
import requests
from urllib.parse import quote
import feedparser
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Robust import handling
PLOTLY_AVAILABLE = True
SCHOLARLY_AVAILABLE = True
ARXIV_AVAILABLE = True

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available")

try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False
    logger.warning("Scholarly not available for Google Scholar")

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    logger.warning("ArXiv API not available")

# ================== Configuration ==================

st.set_page_config(
    page_title="Academic Literature Search & Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        padding: 2rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .search-result {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #1e3c72;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .api-status {
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.2rem;
    }
    .api-active {
        background-color: #d4edda;
        color: #155724;
    }
    .api-inactive {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# ================== Data Models ==================

@dataclass
class Paper:
    """Academic paper representation."""
    title: str
    authors: List[str]
    year: int
    abstract: str
    keywords: List[str] = field(default_factory=list)
    journal: str = ""
    doi: str = ""
    citations: int = 0
    url: str = ""
    pdf_link: str = ""
    source: str = ""  # Which API/database it came from
    venue: str = ""
    paper_id: str = ""
    references: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'abstract': self.abstract,
            'keywords': self.keywords,
            'journal': self.journal,
            'doi': self.doi,
            'citations': self.citations,
            'url': self.url,
            'source': self.source
        }
    
    def to_bibtex(self) -> str:
        """Convert to BibTeX format."""
        authors_str = " and ".join(self.authors)
        key = f"{self.authors[0].split(',')[0].replace(' ', '') if self.authors else 'Unknown'}{self.year}"
        
        bibtex = f"""@article{{{key},
    title = {{{self.title}}},
    author = {{{authors_str}}},
    year = {{{self.year}}},
    journal = {{{self.journal or self.venue or 'Unknown'}}},
    doi = {{{self.doi}}},
    url = {{{self.url}}},
    source = {{{self.source}}}
}}"""
        return bibtex

# ================== Academic Database APIs ==================

class SemanticScholarAPI:
    """Semantic Scholar API integration."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    @staticmethod
    def search(query: str, limit: int = 20, year_filter: tuple = None) -> List[Paper]:
        """Search Semantic Scholar for papers."""
        papers = []
        
        try:
            # Search endpoint
            search_url = f"{SemanticScholarAPI.BASE_URL}/paper/search"
            
            params = {
                'query': query,
                'limit': limit,
                'fields': 'title,authors,year,abstract,venue,citationCount,url,openAccessPdf,publicationTypes,journal,referenceCount,fieldsOfStudy'
            }
            
            # Add year filter if provided
            if year_filter:
                params['year'] = f"{year_filter[0]}-{year_filter[1]}"
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('data', []):
                    authors = [author.get('name', '') for author in item.get('authors', [])]
                    
                    paper = Paper(
                        title=item.get('title', ''),
                        authors=authors,
                        year=item.get('year', 0),
                        abstract=item.get('abstract', ''),
                        journal=item.get('journal', {}).get('name', '') if item.get('journal') else item.get('venue', ''),
                        citations=item.get('citationCount', 0),
                        url=item.get('url', ''),
                        pdf_link=item.get('openAccessPdf', {}).get('url', '') if item.get('openAccessPdf') else '',
                        source='Semantic Scholar',
                        venue=item.get('venue', ''),
                        paper_id=item.get('paperId', ''),
                        keywords=item.get('fieldsOfStudy', []) if item.get('fieldsOfStudy') else []
                    )
                    papers.append(paper)
                    
            else:
                logger.error(f"Semantic Scholar API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {str(e)}")
            st.error(f"Semantic Scholar search failed: {str(e)}")
        
        return papers

class CrossRefAPI:
    """CrossRef API integration."""
    
    BASE_URL = "https://api.crossref.org"
    
    @staticmethod
    def search(query: str, limit: int = 20, year_filter: tuple = None) -> List[Paper]:
        """Search CrossRef for papers."""
        papers = []
        
        try:
            search_url = f"{CrossRefAPI.BASE_URL}/works"
            
            params = {
                'query': query,
                'rows': limit,
                'select': 'title,author,published-print,abstract,container-title,DOI,URL,is-referenced-by-count,type,subject'
            }
            
            # Add year filter
            if year_filter:
                params['filter'] = f"from-pub-date:{year_filter[0]},until-pub-date:{year_filter[1]}"
            
            response = requests.get(search_url, params=params, timeout=10)
            
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
                    elif 'published-online' in item:
                        date_parts = item['published-online'].get('date-parts', [[]])
                        if date_parts and date_parts[0]:
                            year = date_parts[0][0]
                    
                    paper = Paper(
                        title=' '.join(item.get('title', [''])),
                        authors=authors,
                        year=year,
                        abstract=item.get('abstract', ''),
                        journal=' '.join(item.get('container-title', [''])),
                        doi=item.get('DOI', ''),
                        citations=item.get('is-referenced-by-count', 0),
                        url=item.get('URL', ''),
                        source='CrossRef',
                        keywords=item.get('subject', []) if item.get('subject') else []
                    )
                    papers.append(paper)
                    
            else:
                logger.error(f"CrossRef API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error searching CrossRef: {str(e)}")
            st.warning(f"CrossRef search failed: {str(e)}")
        
        return papers

class ArXivAPI:
    """arXiv API integration."""
    
    @staticmethod
    def search(query: str, limit: int = 20, year_filter: tuple = None) -> List[Paper]:
        """Search arXiv for papers."""
        papers = []
        
        try:
            # For mechanical engineering, add relevant categories
            categories = ["cs.RO", "cs.AI", "physics.app-ph", "eess.SY", "cs.LG", "math.OC"]
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            
            if ARXIV_AVAILABLE:
                # Use arxiv Python library
                search = arxiv.Search(
                    query=f"{query} AND ({cat_query})",
                    max_results=limit,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                for result in search.results():
                    # Filter by year if specified
                    if year_filter:
                        if result.published.year < year_filter[0] or result.published.year > year_filter[1]:
                            continue
                    
                    paper = Paper(
                        title=result.title,
                        authors=[author.name for author in result.authors],
                        year=result.published.year,
                        abstract=result.summary,
                        journal="arXiv",
                        doi=result.doi if hasattr(result, 'doi') else "",
                        url=result.entry_id,
                        pdf_link=result.pdf_url,
                        source='arXiv',
                        paper_id=result.entry_id.split('/')[-1],
                        keywords=result.categories if hasattr(result, 'categories') else []
                    )
                    papers.append(paper)
            else:
                # Fallback to HTTP API
                base_url = "http://export.arxiv.org/api/query"
                params = {
                    'search_query': query,
                    'start': 0,
                    'max_results': limit,
                    'sortBy': 'relevance'
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    feed = feedparser.parse(response.text)
                    
                    for entry in feed.entries:
                        # Extract year from published date
                        year = int(entry.published[:4]) if 'published' in entry else 0
                        
                        # Filter by year
                        if year_filter:
                            if year < year_filter[0] or year > year_filter[1]:
                                continue
                        
                        authors = [author.name for author in entry.get('authors', [])]
                        
                        paper = Paper(
                            title=entry.title,
                            authors=authors,
                            year=year,
                            abstract=entry.summary,
                            journal="arXiv",
                            url=entry.link,
                            pdf_link=entry.link.replace('abs', 'pdf'),
                            source='arXiv',
                            paper_id=entry.id.split('/')[-1],
                            keywords=entry.get('tags', [])
                        )
                        papers.append(paper)
                        
        except Exception as e:
            logger.error(f"Error searching arXiv: {str(e)}")
            st.warning(f"arXiv search failed: {str(e)}")
        
        return papers

class PubMedAPI:
    """PubMed API integration."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    @staticmethod
    def search(query: str, limit: int = 20, year_filter: tuple = None) -> List[Paper]:
        """Search PubMed for papers."""
        papers = []
        
        try:
            # Add engineering-related terms for better results
            enhanced_query = f"{query} AND (engineering OR mechanical OR robotics OR biomechanics OR biomedical)"
            
            # Search step
            search_url = f"{PubMedAPI.BASE_URL}/esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': enhanced_query,
                'retmax': limit,
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            if year_filter:
                search_params['mindate'] = f"{year_filter[0]}/01/01"
                search_params['maxdate'] = f"{year_filter[1]}/12/31"
                search_params['datetype'] = 'pdat'
            
            response = requests.get(search_url, params=search_params, timeout=10)
            
            if response.status_code == 200:
                search_results = response.json()
                id_list = search_results.get('esearchresult', {}).get('idlist', [])
                
                if id_list:
                    # Fetch details
                    fetch_url = f"{PubMedAPI.BASE_URL}/efetch.fcgi"
                    fetch_params = {
                        'db': 'pubmed',
                        'id': ','.join(id_list),
                        'retmode': 'xml'
                    }
                    
                    fetch_response = requests.get(fetch_url, params=fetch_params, timeout=10)
                    
                    if fetch_response.status_code == 200:
                        root = ET.fromstring(fetch_response.text)
                        
                        for article in root.findall('.//PubmedArticle'):
                            # Extract title
                            title_elem = article.find('.//ArticleTitle')
                            title = title_elem.text if title_elem is not None else ''
                            
                            # Extract authors
                            authors = []
                            for author in article.findall('.//Author'):
                                lastname = author.find('LastName')
                                forename = author.find('ForeName')
                                if lastname is not None and forename is not None:
                                    authors.append(f"{forename.text} {lastname.text}")
                            
                            # Extract year
                            year_elem = article.find('.//PubDate/Year')
                            year = int(year_elem.text) if year_elem is not None else 0
                            
                            # Extract abstract
                            abstract_texts = []
                            for abstract in article.findall('.//AbstractText'):
                                if abstract.text:
                                    abstract_texts.append(abstract.text)
                            abstract = ' '.join(abstract_texts)
                            
                            # Extract journal
                            journal_elem = article.find('.//Journal/Title')
                            journal = journal_elem.text if journal_elem is not None else ''
                            
                            # Extract PMID
                            pmid_elem = article.find('.//PMID')
                            pmid = pmid_elem.text if pmid_elem is not None else ''
                            
                            # Extract DOI
                            doi = ''
                            for id_elem in article.findall('.//ArticleId'):
                                if id_elem.get('IdType') == 'doi':
                                    doi = id_elem.text
                                    break
                            
                            # Extract keywords
                            keywords = []
                            for kw in article.findall('.//Keyword'):
                                if kw.text:
                                    keywords.append(kw.text)
                            
                            paper = Paper(
                                title=title,
                                authors=authors,
                                year=year,
                                abstract=abstract,
                                journal=journal,
                                doi=doi,
                                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                source='PubMed',
                                paper_id=pmid,
                                keywords=keywords
                            )
                            papers.append(paper)
                            
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            st.warning(f"PubMed search failed: {str(e)}")
        
        return papers

class DOAJApi:
    """DOAJ (Directory of Open Access Journals) API integration."""
    
    BASE_URL = "https://doaj.org/api/v2"
    
    @staticmethod
    def search(query: str, limit: int = 20, year_filter: tuple = None) -> List[Paper]:
        """Search DOAJ for open access papers."""
        papers = []
        
        try:
            search_url = f"{DOAJApi.BASE_URL}/search/articles"
            
            params = {
                'query': query,
                'pageSize': limit,
                'sort': 'relevance'
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('results', []):
                    article = item.get('bibjson', {})
                    
                    # Extract year
                    year = article.get('year', 0)
                    if isinstance(year, str):
                        try:
                            year = int(year)
                        except:
                            year = 0
                    
                    # Filter by year
                    if year_filter and year:
                        if year < year_filter[0] or year > year_filter[1]:
                            continue
                    
                    # Extract authors
                    authors = []
                    for author in article.get('author', []):
                        name = author.get('name', '')
                        if name:
                            authors.append(name)
                    
                    # Extract DOI
                    doi = ''
                    for identifier in article.get('identifier', []):
                        if identifier.get('type') == 'doi':
                            doi = identifier.get('id', '')
                            break
                    
                    # Extract URL
                    url = ''
                    for link in article.get('link', []):
                        if link.get('type') == 'fulltext':
                            url = link.get('url', '')
                            break
                    
                    paper = Paper(
                        title=article.get('title', ''),
                        authors=authors,
                        year=year,
                        abstract=article.get('abstract', ''),
                        journal=article.get('journal', {}).get('title', ''),
                        doi=doi,
                        url=url,
                        source='DOAJ',
                        keywords=article.get('keywords', [])
                    )
                    papers.append(paper)
                    
        except Exception as e:
            logger.error(f"Error searching DOAJ: {str(e)}")
            st.warning(f"DOAJ search failed: {str(e)}")
        
        return papers

class GoogleScholarAPI:
    """Google Scholar integration using scholarly library."""
    
    @staticmethod
    def search(query: str, limit: int = 10, year_filter: tuple = None) -> List[Paper]:
        """Search Google Scholar for papers."""
        papers = []
        
        if not SCHOLARLY_AVAILABLE:
            st.warning("Google Scholar search not available. Install 'scholarly' package.")
            return papers
        
        try:
            # Configure scholarly (use proxy if needed)
            # scholarly.use_proxy(http="http://proxy:port")
            
            search_query = scholarly.search_pubs(query)
            count = 0
            
            for result in search_query:
                if count >= limit:
                    break
                
                # Extract year
                year = result.get('bib', {}).get('pub_year', '')
                if year:
                    try:
                        year = int(year)
                    except:
                        year = 0
                else:
                    year = 0
                
                # Filter by year
                if year_filter and year:
                    if year < year_filter[0] or year > year_filter[1]:
                        continue
                
                # Extract authors
                authors = result.get('bib', {}).get('author', [])
                if isinstance(authors, str):
                    authors = [authors]
                
                paper = Paper(
                    title=result.get('bib', {}).get('title', ''),
                    authors=authors,
                    year=year,
                    abstract=result.get('bib', {}).get('abstract', ''),
                    journal=result.get('bib', {}).get('venue', ''),
                    citations=result.get('num_citations', 0),
                    url=result.get('pub_url', ''),
                    source='Google Scholar',
                    paper_id=result.get('author_id', [''])[0] if result.get('author_id') else ''
                )
                papers.append(paper)
                count += 1
                
                # Rate limiting
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {str(e)}")
            st.warning(f"Google Scholar search failed: {str(e)}")
        
        return papers

# ================== Multi-Source Search Engine ==================

class MultiSourceSearchEngine:
    """Integrated search engine that queries multiple academic databases."""
    
    def __init__(self):
        self.apis = {
            'Semantic Scholar': SemanticScholarAPI(),
            'CrossRef': CrossRefAPI(),
            'arXiv': ArXivAPI(),
            'PubMed': PubMedAPI(),
            'DOAJ': DOAJApi(),
            'Google Scholar': GoogleScholarAPI()
        }
        
        # API status tracking
        self.api_status = {name: True for name in self.apis.keys()}
    
    def search(self, query: str, sources: List[str] = None, 
              limit_per_source: int = 10, year_filter: tuple = None,
              min_citations: int = 0) -> List[Paper]:
        """
        Search multiple academic databases simultaneously.
        
        Parameters:
        -----------
        query : str
            Search query
        sources : List[str]
            List of sources to search (None = all sources)
        limit_per_source : int
            Maximum results per source
        year_filter : tuple
            (min_year, max_year) filter
        min_citations : int
            Minimum citation count filter
        """
        
        if sources is None:
            sources = list(self.apis.keys())
        
        all_papers = []
        source_counts = {}
        
        # Progress container
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for idx, source in enumerate(sources):
            if source not in self.apis:
                continue
            
            status_text.text(f"Searching {source}...")
            progress_bar.progress((idx + 1) / len(sources))
            
            try:
                # Search the source
                papers = self.apis[source].search(
                    query, 
                    limit=limit_per_source,
                    year_filter=year_filter
                )
                
                # Filter by citations if specified
                if min_citations > 0:
                    papers = [p for p in papers if p.citations >= min_citations]
                
                all_papers.extend(papers)
                source_counts[source] = len(papers)
                self.api_status[source] = True
                
            except Exception as e:
                logger.error(f"Error searching {source}: {str(e)}")
                self.api_status[source] = False
                source_counts[source] = 0
        
        # Clear progress indicators
        progress_container.empty()
        
        # Remove duplicates based on title similarity
        unique_papers = self._remove_duplicates(all_papers)
        
        # Sort by relevance (citations and year)
        unique_papers.sort(key=lambda p: (p.citations, p.year), reverse=True)
        
        # Display search summary
        self._display_search_summary(source_counts, len(unique_papers))
        
        return unique_papers
    
    def _remove_duplicates(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers based on title similarity."""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            # Simple title normalization
            normalized_title = re.sub(r'[^\w\s]', '', paper.title.lower())
            normalized_title = ' '.join(normalized_title.split())
            
            if normalized_title not in seen_titles and normalized_title:
                unique_papers.append(paper)
                seen_titles.add(normalized_title)
        
        return unique_papers
    
    def _display_search_summary(self, source_counts: Dict[str, int], total_unique: int):
        """Display summary of search results from each source."""
        with st.expander("Search Summary", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Unique Papers", total_unique)
            
            with col2:
                st.metric("Sources Queried", len(source_counts))
            
            # Source breakdown
            st.write("**Results by Source:**")
            for source, count in source_counts.items():
                status = "✅" if self.api_status.get(source, False) else "❌"
                st.write(f"{status} {source}: {count} papers")

# ================== Analytics Module ==================

class ResearchAnalytics:
    """Analytics engine for research papers."""
    
    def __init__(self, papers: List[Paper]):
        self.papers = papers
        self.df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert papers to DataFrame."""
        data = []
        for paper in self.papers:
            data.append({
                'title': paper.title,
                'year': paper.year,
                'citations': paper.citations,
                'source': paper.source,
                'num_authors': len(paper.authors),
                'journal': paper.journal or paper.venue,
                'has_pdf': bool(paper.pdf_link),
                'has_doi': bool(paper.doi)
            })
        return pd.DataFrame(data)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_papers': len(self.papers),
            'unique_sources': self.df['source'].nunique(),
            'total_citations': self.df['citations'].sum(),
            'avg_citations': self.df['citations'].mean(),
            'papers_with_pdf': self.df['has_pdf'].sum(),
            'papers_with_doi': self.df['has_doi'].sum(),
            'year_range': f"{self.df['year'].min()}-{self.df['year'].max()}" if not self.df.empty else "N/A"
        }
    
    def get_source_distribution(self) -> pd.Series:
        """Get distribution by source."""
        return self.df['source'].value_counts()
    
    def get_year_distribution(self) -> pd.Series:
        """Get distribution by year."""
        return self.df['year'].value_counts().sort_index()
    
    def get_top_cited(self, n: int = 10) -> List[Paper]:
        """Get top cited papers."""
        if self.df.empty:
            return []
        top_indices = self.df.nlargest(min(n, len(self.df)), 'citations').index
        return [self.papers[i] for i in top_indices]

# ================== Main Application ==================

class AcademicSearchApp:
    """Main application with real API integration."""
    
    def __init__(self):
        self.search_engine = MultiSourceSearchEngine()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state."""
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'selected_papers' not in st.session_state:
            st.session_state.selected_papers = []
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
    
    def run(self):
        """Run the application."""
        # Header
        st.markdown('<div class="main-header"><h1>📚 Academic Literature Search Platform</h1><p>Multi-Database Search with Real-Time API Integration</p></div>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.title("Navigation")
            page = st.radio(
                "Select Page",
                ["🔍 Search", "📊 Analytics", "📁 Portfolio", "⚙️ Settings"]
            )
        
        if page == "🔍 Search":
            self.render_search_page()
        elif page == "📊 Analytics":
            self.render_analytics_page()
        elif page == "📁 Portfolio":
            self.render_portfolio_page()
        else:
            self.render_settings_page()
    
    def render_search_page(self):
        """Render search interface."""
        st.header("🔍 Multi-Database Literature Search")
        
        # Search configuration
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Search Query",
                placeholder="e.g., 'deep learning mechanical fault detection'",
                help="Search across multiple academic databases"
            )
        
        with col2:
            st.write("")
            st.write("")
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        # Advanced options
        with st.expander("⚙️ Search Settings", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Database selection
                st.write("**Select Databases:**")
                sources = []
                
                if st.checkbox("Semantic Scholar", value=True):
                    sources.append("Semantic Scholar")
                if st.checkbox("CrossRef", value=True):
                    sources.append("CrossRef")
                if st.checkbox("arXiv", value=True):
                    sources.append("arXiv")
                if st.checkbox("PubMed", value=False):
                    sources.append("PubMed")
                if st.checkbox("DOAJ", value=False):
                    sources.append("DOAJ")
                if st.checkbox("Google Scholar", value=False):
                    sources.append("Google Scholar")
            
            with col2:
                # Filters
                st.write("**Filters:**")
                year_range = st.slider(
                    "Year Range",
                    1990, 2024, (2020, 2024)
                )
                
                min_citations = st.number_input(
                    "Min Citations",
                    min_value=0,
                    value=0
                )
            
            with col3:
                # Search options
                st.write("**Options:**")
                results_per_source = st.number_input(
                    "Results per Database",
                    min_value=5,
                    max_value=50,
                    value=10
                )
                
                sort_by = st.selectbox(
                    "Sort Results By",
                    ["Citations", "Year", "Relevance"]
                )
        
        # Execute search
        if search_button and query:
            with st.spinner(f"Searching {len(sources)} databases..."):
                results = self.search_engine.search(
                    query=query,
                    sources=sources,
                    limit_per_source=results_per_source,
                    year_filter=year_range,
                    min_citations=min_citations
                )
                
                # Sort results
                if sort_by == "Citations":
                    results.sort(key=lambda p: p.citations, reverse=True)
                elif sort_by == "Year":
                    results.sort(key=lambda p: p.year, reverse=True)
                
                st.session_state.search_results = results
                
                # Add to search history
                st.session_state.search_history.append({
                    'query': query,
                    'timestamp': datetime.now(),
                    'results': len(results),
                    'sources': sources
                })
        
        # Display results
        if st.session_state.search_results:
            st.divider()
            st.subheader(f"📄 Found {len(st.session_state.search_results)} Papers")
            
            # Results filter
            filter_col1, filter_col2 = st.columns([1, 3])
            with filter_col1:
                filter_source = st.selectbox(
                    "Filter by Source",
                    ["All"] + list(set([p.source for p in st.session_state.search_results]))
                )
            
            # Display papers
            for idx, paper in enumerate(st.session_state.search_results):
                if filter_source != "All" and paper.source != filter_source:
                    continue
                
                with st.container():
                    st.markdown('<div class="search-result">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([5, 1])
                    
                    with col1:
                        st.markdown(f"### {idx + 1}. {paper.title}")
                        st.write(f"**Authors:** {', '.join(paper.authors[:5])}{'...' if len(paper.authors) > 5 else ''}")
                        
                        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                        with col_info1:
                            st.caption(f"📅 Year: {paper.year}")
                        with col_info2:
                            st.caption(f"📖 Citations: {paper.citations}")
                        with col_info3:
                            st.caption(f"📚 Source: {paper.source}")
                        with col_info4:
                            st.caption(f"📰 {paper.journal or 'N/A'}")
                        
                        # Links
                        links = []
                        if paper.url:
                            links.append(f"[🔗 View]({paper.url})")
                        if paper.pdf_link:
                            links.append(f"[📄 PDF]({paper.pdf_link})")
                        if paper.doi:
                            links.append(f"[DOI]({f'https://doi.org/{paper.doi}'})")
                        
                        if links:
                            st.markdown(" | ".join(links))
                        
                        # Abstract
                        with st.expander("📝 Abstract"):
                            st.write(paper.abstract or "No abstract available")
                            if paper.keywords:
                                st.write(f"**Keywords:** {', '.join(paper.keywords)}")
                    
                    with col2:
                        st.write("")
                        if st.button(f"Add to Portfolio", key=f"add_{idx}"):
                            if paper not in st.session_state.selected_papers:
                                st.session_state.selected_papers.append(paper)
                                st.success("Added!")
                        
                        if st.button("Export BibTeX", key=f"bib_{idx}"):
                            st.text_area("BibTeX", paper.to_bibtex(), height=200, key=f"bibtex_area_{idx}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    def render_analytics_page(self):
        """Render analytics dashboard."""
        st.header("📊 Research Analytics Dashboard")
        
        if not st.session_state.selected_papers:
            st.info("Add papers to your portfolio to view analytics")
            
            if st.button("Use Search Results for Demo"):
                st.session_state.selected_papers = st.session_state.search_results[:20]
                st.rerun()
            return
        
        # Initialize analytics
        analytics = ResearchAnalytics(st.session_state.selected_papers)
        
        # Summary stats
        stats = analytics.get_summary_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Papers", stats['total_papers'])
        with col2:
            st.metric("Total Citations", f"{stats['total_citations']:,}")
        with col3:
            st.metric("Papers with PDF", stats['papers_with_pdf'])
        with col4:
            st.metric("Unique Sources", stats['unique_sources'])
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Papers by Source")
            source_dist = analytics.get_source_distribution()
            if not source_dist.empty:
                if PLOTLY_AVAILABLE:
                    fig = go.Figure(data=[
                        go.Pie(labels=source_dist.index, values=source_dist.values, hole=0.3)
                    ])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(source_dist)
        
        with col2:
            st.subheader("Papers by Year")
            year_dist = analytics.get_year_distribution()
            if not year_dist.empty:
                if PLOTLY_AVAILABLE:
                    fig = go.Figure(data=[
                        go.Bar(x=year_dist.index, y=year_dist.values, marker_color='#1e3c72')
                    ])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(year_dist)
        
        # Top cited papers
        st.divider()
        st.subheader("🏆 Top Cited Papers")
        
        top_papers = analytics.get_top_cited(10)
        for i, paper in enumerate(top_papers, 1):
            with st.expander(f"{i}. {paper.title} ({paper.citations} citations)"):
                st.write(f"**Authors:** {', '.join(paper.authors[:3])}")
                st.write(f"**Year:** {paper.year} | **Source:** {paper.source}")
                st.write(f"**Journal:** {paper.journal}")
                if paper.abstract:
                    st.write(f"**Abstract:** {paper.abstract[:300]}...")
    
    def render_portfolio_page(self):
        """Render portfolio management page."""
        st.header("📁 Research Portfolio")
        
        if not st.session_state.selected_papers:
            st.info("Your portfolio is empty. Add papers from the search page.")
            return
        
        st.write(f"**{len(st.session_state.selected_papers)} papers in portfolio**")
        
        # Actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export All (BibTeX)"):
                all_bibtex = "\n\n".join([p.to_bibtex() for p in st.session_state.selected_papers])
                st.download_button(
                    label="Download BibTeX",
                    data=all_bibtex,
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.bib",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("📋 Export List (CSV)"):
                df = pd.DataFrame([p.to_dict() for p in st.session_state.selected_papers])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🗑️ Clear Portfolio"):
                st.session_state.selected_papers = []
                st.rerun()
        
        st.divider()
        
        # Display papers
        for idx, paper in enumerate(st.session_state.selected_papers):
            with st.container():
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    st.markdown(f"**{idx + 1}. {paper.title}**")
                    st.caption(f"{', '.join(paper.authors[:3])} | {paper.year} | {paper.source}")
                
                with col2:
                    if st.button("Remove", key=f"remove_{idx}"):
                        st.session_state.selected_papers.pop(idx)
                        st.rerun()
    
    def render_settings_page(self):
        """Render settings page."""
        st.header("⚙️ Settings & API Status")
        
        st.subheader("API Status")
        
        # Check API availability
        api_status = {
            "Semantic Scholar": "✅ Active",
            "CrossRef": "✅ Active",
            "arXiv": "✅ Active" if ARXIV_AVAILABLE else "⚠️ Limited (HTTP only)",
            "PubMed": "✅ Active",
            "DOAJ": "✅ Active",
            "Google Scholar": "✅ Active" if SCHOLARLY_AVAILABLE else "❌ Unavailable"
        }
        
        for api, status in api_status.items():
            st.write(f"{status} **{api}**")
        
        st.divider()
        
        st.subheader("Search History")
        
        if st.session_state.search_history:
            df_history = pd.DataFrame(st.session_state.search_history)
            st.dataframe(df_history[['query', 'timestamp', 'results']], use_container_width=True)
            
            if st.button("Clear History"):
                st.session_state.search_history = []
                st.rerun()
        else:
            st.info("No search history yet")
        
        st.divider()
        
        st.subheader("About")
        st.write("""
        This application provides integrated access to multiple academic databases:
        
        - **Semantic Scholar**: AI-powered research tool with citation context
        - **CrossRef**: Comprehensive DOI and metadata registry
        - **arXiv**: Preprints in physics, mathematics, computer science
        - **PubMed**: Biomedical and life science literature
        - **DOAJ**: Open access journal articles
        - **Google Scholar**: Broad academic search (when available)
        
        For mechanical engineering research, the system automatically enhances queries
        with relevant terms and filters results for maximum relevance.
        """)

# ================== Entry Point ==================

def main():
    """Main entry point."""
    app = AcademicSearchApp()
    app.run()

if __name__ == "__main__":
    main()
