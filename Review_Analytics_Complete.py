"""
Integrated Mechanical Engineering Literature Search and Analytics Application
Version 3.1 - With Robust Import Handling and Dependency Management

This version includes proper error handling for missing dependencies and
provides fallback visualization options when certain libraries are unavailable.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
import base64
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Robust import handling with fallback options
PLOTLY_AVAILABLE = True
NETWORKX_AVAILABLE = True
WORDCLOUD_AVAILABLE = True

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    logger.info("Plotly successfully imported")
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Using alternative visualization methods.")
    # Fallback to matplotlib if available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        logger.info("Matplotlib available as fallback")
    except ImportError:
        logger.warning("Matplotlib also not available")

try:
    import networkx as nx
    logger.info("NetworkX successfully imported")
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available. Network analysis features disabled.")

try:
    from wordcloud import WordCloud
    logger.info("WordCloud successfully imported")
except ImportError:
    WORDCLOUD_AVAILABLE = False
    logger.warning("WordCloud not available. Word cloud features disabled.")

# ================== Configuration ==================

st.set_page_config(
    page_title="Literature Search & Analytics Platform",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: #f7f7f7;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .search-result {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ================== Data Models ==================

@dataclass
class Paper:
    """Academic paper representation with comprehensive metadata."""
    title: str
    authors: List[str]
    year: int
    abstract: str
    keywords: List[str]
    journal: str = ""
    doi: str = ""
    citations: int = 0
    references: List[str] = field(default_factory=list)
    url: str = ""
    pdf_link: str = ""
    impact_factor: float = 0.0
    field_of_study: List[str] = field(default_factory=list)
    methodology: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    
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
            'url': self.url
        }
    
    def to_bibtex(self) -> str:
        """Convert paper to BibTeX format."""
        authors_str = " and ".join(self.authors)
        key = f"{self.authors[0].split(',')[0] if self.authors else 'Unknown'}{self.year}"
        
        bibtex = f"""@article{{{key},
    title = {{{self.title}}},
    author = {{{authors_str}}},
    year = {{{self.year}}},
    journal = {{{self.journal}}},
    doi = {{{self.doi}}},
    abstract = {{{self.abstract[:200]}}},
    keywords = {{{", ".join(self.keywords)}}}
}}"""
        return bibtex

# ================== Search Engine ==================

class EnhancedSearchEngine:
    """Enhanced search engine with multiple search strategies."""
    
    def __init__(self):
        self.papers = self._load_sample_database()
        self.index = self._build_search_index()
    
    def _load_sample_database(self) -> List[Paper]:
        """Load sample papers for demonstration."""
        papers = [
            Paper(
                title="Deep Learning Applications in Mechanical Fault Detection: A Comprehensive Survey",
                authors=["Zhang, Wei", "Li, Xiang", "Wang, Shuai", "Chen, Ming"],
                year=2024,
                abstract="This comprehensive survey examines the application of deep learning techniques in mechanical fault detection systems. We review recent advances in convolutional neural networks, recurrent neural networks, and transformer models for diagnosing faults in rotating machinery, analyzing over 200 papers published between 2018 and 2024.",
                keywords=["deep learning", "fault detection", "mechanical systems", "neural networks", "predictive maintenance"],
                journal="Mechanical Systems and Signal Processing",
                doi="10.1016/j.ymssp.2024.110234",
                citations=45,
                impact_factor=6.8,
                field_of_study=["Mechanical Engineering", "Artificial Intelligence", "Signal Processing"],
                methodology=["Survey", "Systematic Review", "Meta-analysis"]
            ),
            Paper(
                title="Advances in Composite Materials for Aerospace Applications: Manufacturing and Design",
                authors=["Johnson, Michael A.", "Smith, Robert T.", "Chen, Liu", "Anderson, Karen"],
                year=2023,
                abstract="Recent developments in composite materials have revolutionized aerospace engineering. This paper presents novel manufacturing techniques for carbon fiber reinforced polymers, including automated fiber placement and resin transfer molding optimizations.",
                keywords=["composite materials", "aerospace", "carbon fiber", "manufacturing", "structural optimization"],
                journal="Composites Part A: Applied Science and Manufacturing",
                doi="10.1016/j.compositesa.2023.107234",
                citations=78,
                impact_factor=7.2,
                field_of_study=["Materials Science", "Aerospace Engineering", "Manufacturing"],
                methodology=["Experimental", "Computational Modeling", "Case Study"]
            ),
            Paper(
                title="Machine Learning for Predictive Maintenance in Industry 4.0: A Systematic Review",
                authors=["Wang, Hong", "Thompson, David", "Kumar, Amit", "Garcia, Maria"],
                year=2023,
                abstract="Predictive maintenance using machine learning has become crucial for modern manufacturing systems. This systematic review analyzes 150 studies on ML applications in predictive maintenance, focusing on algorithm performance, implementation challenges, and industrial case studies.",
                keywords=["predictive maintenance", "machine learning", "Industry 4.0", "smart manufacturing", "condition monitoring"],
                journal="Journal of Manufacturing Systems",
                doi="10.1016/j.jmsy.2023.05.012",
                citations=92,
                impact_factor=8.6,
                field_of_study=["Manufacturing Engineering", "Data Science", "Industrial Engineering"],
                methodology=["Systematic Review", "Meta-analysis", "Case Study Analysis"]
            ),
            Paper(
                title="Optimization of Wind Turbine Blade Design Using Genetic Algorithms and CFD",
                authors=["Anderson, Keith", "Liu, Yang", "Martinez, Juan", "Brown, Sarah"],
                year=2024,
                abstract="This study presents a novel approach to wind turbine blade optimization combining genetic algorithms with computational fluid dynamics simulations. The proposed method achieves 15% improvement in energy capture efficiency.",
                keywords=["wind energy", "optimization", "genetic algorithms", "CFD", "renewable energy"],
                journal="Renewable Energy",
                doi="10.1016/j.renene.2024.119234",
                citations=23,
                impact_factor=8.1,
                field_of_study=["Renewable Energy", "Mechanical Engineering", "Computational Methods"],
                methodology=["Optimization", "Simulation", "Experimental Validation"]
            ),
            Paper(
                title="Thermal Management in Electric Vehicle Battery Systems: Challenges and Solutions",
                authors=["Park, Ji-won", "Wilson, Emma", "Zhang, Qiang", "Lee, Sung-ho"],
                year=2024,
                abstract="Effective thermal management is critical for electric vehicle battery performance and safety. This paper reviews current thermal management strategies, presents novel cooling system designs, and validates them through experimental testing.",
                keywords=["electric vehicles", "battery thermal management", "cooling systems", "heat transfer", "energy storage"],
                journal="Applied Thermal Engineering",
                doi="10.1016/j.applthermaleng.2024.122345",
                citations=56,
                impact_factor=5.4,
                field_of_study=["Thermal Engineering", "Automotive Engineering", "Energy Systems"],
                methodology=["Review", "Experimental", "Numerical Simulation"]
            ),
            Paper(
                title="Additive Manufacturing of Metal Matrix Composites: Processing and Properties",
                authors=["Rodriguez, Carlos", "Wang, Mei", "Schmidt, Hans", "Patel, Raj"],
                year=2023,
                abstract="This research investigates additive manufacturing techniques for producing metal matrix composites with enhanced mechanical properties. We demonstrate successful fabrication of Al-SiC composites using selective laser melting.",
                keywords=["additive manufacturing", "metal matrix composites", "3D printing", "selective laser melting", "materials processing"],
                journal="Materials Science and Engineering: A",
                doi="10.1016/j.msea.2023.145678",
                citations=67,
                impact_factor=6.0,
                field_of_study=["Materials Science", "Manufacturing", "Additive Manufacturing"],
                methodology=["Experimental", "Characterization", "Mechanical Testing"]
            )
        ]
        return papers
    
    def _build_search_index(self) -> Dict[str, List[int]]:
        """Build inverted index for efficient searching."""
        index = defaultdict(list)
        
        for idx, paper in enumerate(self.papers):
            # Index title
            for word in paper.title.lower().split():
                word = re.sub(r'[^\w\s]', '', word)
                if word:
                    index[word].append(idx)
            
            # Index abstract
            for word in paper.abstract.lower().split():
                word = re.sub(r'[^\w\s]', '', word)
                if word:
                    index[word].append(idx)
            
            # Index keywords
            for keyword in paper.keywords:
                for word in keyword.lower().split():
                    index[word].append(idx)
            
            # Index authors
            for author in paper.authors:
                name_parts = author.lower().replace(',', '').split()
                for part in name_parts:
                    index[part].append(idx)
        
        return dict(index)
    
    def search(self, query: str, filters: Dict[str, Any] = None) -> List[Paper]:
        """Execute search with optional filters."""
        if not query and not filters:
            return self.papers
        
        results = []
        query_words = [re.sub(r'[^\w\s]', '', word.lower()) for word in query.split()] if query else []
        
        # Score papers based on query
        paper_scores = Counter()
        for word in query_words:
            if word in self.index:
                for idx in self.index[word]:
                    paper_scores[idx] += 1
        
        # If no query, include all papers
        if not query_words:
            paper_scores = {i: 1 for i in range(len(self.papers))}
        
        # Apply filters and collect results
        for idx, score in paper_scores.items():
            paper = self.papers[idx]
            
            # Apply filters
            if filters:
                if filters.get('year_min') and paper.year < filters['year_min']:
                    continue
                if filters.get('year_max') and paper.year > filters['year_max']:
                    continue
                if filters.get('min_citations') and paper.citations < filters['min_citations']:
                    continue
                if filters.get('journals') and paper.journal not in filters['journals']:
                    continue
                if filters.get('fields') and not any(f in paper.field_of_study for f in filters['fields']):
                    continue
            
            results.append((paper, score))
        
        # Sort by relevance score
        results.sort(key=lambda x: (x[1], x[0].citations), reverse=True)
        
        return [paper for paper, _ in results]

# ================== Analytics Engine ==================

class AnalyticsEngine:
    """Comprehensive analytics engine for research papers."""
    
    def __init__(self, papers: List[Paper]):
        self.papers = papers
        self.df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert papers to DataFrame for analysis."""
        data = []
        for paper in self.papers:
            data.append({
                'title': paper.title,
                'year': paper.year,
                'citations': paper.citations,
                'num_authors': len(paper.authors),
                'num_keywords': len(paper.keywords),
                'journal': paper.journal,
                'impact_factor': paper.impact_factor,
                'first_author': paper.authors[0] if paper.authors else "",
                'fields': ', '.join(paper.field_of_study),
                'methods': ', '.join(paper.methodology)
            })
        return pd.DataFrame(data)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        return {
            'total_papers': len(self.papers),
            'total_citations': self.df['citations'].sum(),
            'avg_citations': self.df['citations'].mean(),
            'median_citations': self.df['citations'].median(),
            'avg_authors': self.df['num_authors'].mean(),
            'year_range': f"{self.df['year'].min()} - {self.df['year'].max()}",
            'unique_journals': self.df['journal'].nunique(),
            'avg_impact_factor': self.df['impact_factor'].mean()
        }
    
    def get_temporal_distribution(self) -> pd.Series:
        """Get publication distribution by year."""
        return self.df['year'].value_counts().sort_index()
    
    def get_top_cited_papers(self, n: int = 5) -> List[Paper]:
        """Get top cited papers."""
        top_indices = self.df.nlargest(n, 'citations').index
        return [self.papers[i] for i in top_indices]
    
    def get_keyword_frequency(self) -> Counter:
        """Analyze keyword frequency."""
        all_keywords = []
        for paper in self.papers:
            all_keywords.extend(paper.keywords)
        return Counter(all_keywords)
    
    def get_author_statistics(self) -> Dict[str, Any]:
        """Calculate author-related statistics."""
        all_authors = []
        for paper in self.papers:
            all_authors.extend(paper.authors)
        
        author_counts = Counter(all_authors)
        
        return {
            'total_unique_authors': len(set(all_authors)),
            'most_prolific': author_counts.most_common(5),
            'collaboration_index': self.df['num_authors'].mean(),
            'single_author_papers': len(self.df[self.df['num_authors'] == 1])
        }
    
    def get_journal_distribution(self) -> pd.Series:
        """Get distribution of papers by journal."""
        return self.df['journal'].value_counts()
    
    def get_methodology_distribution(self) -> Counter:
        """Analyze research methodology distribution."""
        all_methods = []
        for paper in self.papers:
            all_methods.extend(paper.methodology)
        return Counter(all_methods)

# ================== Visualization Functions ==================

class VisualizationManager:
    """Manages visualizations with fallback options."""
    
    @staticmethod
    def create_bar_chart(data: pd.Series, title: str, x_label: str, y_label: str):
        """Create bar chart with Plotly or matplotlib fallback."""
        if PLOTLY_AVAILABLE:
            fig = go.Figure(data=[
                go.Bar(x=data.index, y=data.values, marker_color='#667eea')
            ])
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                template='plotly_white'
            )
            return fig
        else:
            # Fallback to matplotlib/native Streamlit
            st.bar_chart(data)
            st.caption(f"{title}")
            return None
    
    @staticmethod
    def create_line_chart(data: pd.Series, title: str):
        """Create line chart with Plotly or matplotlib fallback."""
        if PLOTLY_AVAILABLE:
            fig = go.Figure(data=[
                go.Scatter(x=data.index, y=data.values, mode='lines+markers',
                          line_color='#667eea', marker=dict(size=8))
            ])
            fig.update_layout(
                title=title,
                xaxis_title='Year',
                yaxis_title='Count',
                template='plotly_white'
            )
            return fig
        else:
            st.line_chart(data)
            st.caption(f"{title}")
            return None
    
    @staticmethod
    def create_pie_chart(data: pd.Series, title: str):
        """Create pie chart with Plotly or matplotlib fallback."""
        if PLOTLY_AVAILABLE:
            fig = go.Figure(data=[
                go.Pie(labels=data.index, values=data.values, hole=0.3)
            ])
            fig.update_layout(title=title)
            return fig
        else:
            # Simple table fallback
            st.write(f"**{title}**")
            st.dataframe(data)
            return None
    
    @staticmethod
    def create_scatter_plot(df: pd.DataFrame, x: str, y: str, title: str):
        """Create scatter plot with Plotly or matplotlib fallback."""
        if PLOTLY_AVAILABLE:
            fig = px.scatter(df, x=x, y=y, title=title,
                           hover_data=['title'], color='impact_factor',
                           color_continuous_scale='Viridis')
            return fig
        else:
            st.write(f"**{title}**")
            st.scatter_chart(df[[x, y]])
            return None
    
    @staticmethod
    def create_word_cloud(word_freq: Dict[str, int]):
        """Create word cloud if available, otherwise show frequency table."""
        if WORDCLOUD_AVAILABLE:
            wordcloud = WordCloud(width=800, height=400,
                                 background_color='white').generate_from_frequencies(word_freq)
            return wordcloud
        else:
            st.write("**Keyword Frequency**")
            df_freq = pd.DataFrame(list(word_freq.items()), columns=['Keyword', 'Frequency'])
            df_freq = df_freq.sort_values('Frequency', ascending=False).head(20)
            st.dataframe(df_freq)
            return None

# ================== Main Application ==================

class LiteratureSearchApp:
    """Main application class."""
    
    def __init__(self):
        self.search_engine = EnhancedSearchEngine()
        self.viz_manager = VisualizationManager()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables."""
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'selected_papers' not in st.session_state:
            st.session_state.selected_papers = []
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
    
    def run(self):
        """Run the main application."""
        # Header
        st.markdown('<div class="main-header"><h1>📚 Academic Literature Search & Analytics Platform</h1><p>Advanced Research Tool for Mechanical Engineering</p></div>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.title("Navigation")
            page = st.radio(
                "Select Module",
                ["🔍 Literature Search", "📊 Analytics Dashboard", "📁 Research Portfolio", "ℹ️ About"]
            )
        
        # Route to appropriate page
        if page == "🔍 Literature Search":
            self.render_search_page()
        elif page == "📊 Analytics Dashboard":
            self.render_analytics_page()
        elif page == "📁 Research Portfolio":
            self.render_portfolio_page()
        else:
            self.render_about_page()
    
    def render_search_page(self):
        """Render the search interface."""
        st.header("🔍 Literature Search")
        
        # Search input
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Enter your search query",
                placeholder="e.g., 'deep learning fault detection'",
                help="Search across titles, abstracts, keywords, and authors"
            )
        
        with col2:
            st.write("")
            st.write("")
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        # Filters
        with st.expander("Advanced Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                year_range = st.slider("Year Range", 2020, 2024, (2022, 2024))
                min_citations = st.number_input("Minimum Citations", 0, 1000, 0)
            
            with col2:
                available_journals = list(set([p.journal for p in self.search_engine.papers]))
                selected_journals = st.multiselect("Journals", available_journals)
            
            with col3:
                available_fields = list(set(sum([p.field_of_study for p in self.search_engine.papers], [])))
                selected_fields = st.multiselect("Fields of Study", available_fields)
        
        # Execute search
        if search_button or query:
            filters = {
                'year_min': year_range[0],
                'year_max': year_range[1],
                'min_citations': min_citations,
                'journals': selected_journals if selected_journals else None,
                'fields': selected_fields if selected_fields else None
            }
            
            with st.spinner("Searching..."):
                results = self.search_engine.search(query, filters)
                st.session_state.search_results = results
                
                if query:
                    st.session_state.search_history.append({
                        'query': query,
                        'timestamp': datetime.now(),
                        'results_count': len(results)
                    })
            
            st.success(f"Found {len(results)} papers")
        
        # Display results
        if st.session_state.search_results:
            st.divider()
            st.subheader("Search Results")
            
            for i, paper in enumerate(st.session_state.search_results):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**{i+1}. {paper.title}**")
                        st.caption(f"Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
                        st.caption(f"📅 {paper.year} | 📚 {paper.journal} | 📖 {paper.citations} citations")
                        
                        with st.expander("View Abstract"):
                            st.write(paper.abstract)
                            st.write(f"**Keywords:** {', '.join(paper.keywords)}")
                            st.write(f"**DOI:** {paper.doi}")
                    
                    with col2:
                        if st.button(f"Add to Portfolio", key=f"add_{i}"):
                            if paper not in st.session_state.selected_papers:
                                st.session_state.selected_papers.append(paper)
                                st.success("Added!")
                        
                        if st.button(f"Export BibTeX", key=f"bib_{i}"):
                            bibtex = paper.to_bibtex()
                            st.text_area("BibTeX", bibtex, height=150, key=f"bibtex_{i}")
                    
                    st.divider()
    
    def render_analytics_page(self):
        """Render the analytics dashboard."""
        st.header("📊 Analytics Dashboard")
        
        if not st.session_state.selected_papers:
            st.info("📌 Please add papers to your portfolio from the search page to view analytics.")
            
            # Show demo analytics with all papers
            if st.button("Show Demo Analytics with Sample Data"):
                st.session_state.selected_papers = self.search_engine.papers
                st.rerun()
            return
        
        # Initialize analytics
        analytics = AnalyticsEngine(st.session_state.selected_papers)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        stats = analytics.get_summary_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Papers", stats['total_papers'])
        with col2:
            st.metric("Total Citations", stats['total_citations'])
        with col3:
            st.metric("Avg Citations", f"{stats['avg_citations']:.1f}")
        with col4:
            st.metric("Avg Impact Factor", f"{stats['avg_impact_factor']:.2f}")
        
        st.divider()
        
        # Visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Temporal Analysis",
            "👥 Author Analysis",
            "🔑 Keyword Analysis",
            "📚 Journal Analysis",
            "🔬 Methodology Analysis"
        ])
        
        with tab1:
            st.subheader("Publication Trends Over Time")
            temporal_dist = analytics.get_temporal_distribution()
            if not temporal_dist.empty:
                fig = self.viz_manager.create_line_chart(temporal_dist, "Publications per Year")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Citations over time
            st.subheader("Citation Analysis")
            citation_by_year = analytics.df.groupby('year')['citations'].sum()
            fig = self.viz_manager.create_bar_chart(
                citation_by_year, "Citations by Year", "Year", "Total Citations"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Author Statistics")
            author_stats = analytics.get_author_statistics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Unique Authors", author_stats['total_unique_authors'])
                st.metric("Avg Authors per Paper", f"{author_stats['collaboration_index']:.2f}")
            
            with col2:
                st.metric("Single Author Papers", author_stats['single_author_papers'])
            
            if author_stats['most_prolific']:
                st.subheader("Most Prolific Authors")
                author_df = pd.DataFrame(author_stats['most_prolific'], 
                                        columns=['Author', 'Papers'])
                st.dataframe(author_df, use_container_width=True)
        
        with tab3:
            st.subheader("Keyword Analysis")
            keyword_freq = analytics.get_keyword_frequency()
            
            if keyword_freq:
                # Show top keywords
                top_keywords = dict(keyword_freq.most_common(15))
                
                if WORDCLOUD_AVAILABLE:
                    st.subheader("Keyword Cloud")
                    wordcloud = self.viz_manager.create_word_cloud(top_keywords)
                    if wordcloud:
                        st.image(wordcloud.to_array())
                else:
                    self.viz_manager.create_word_cloud(top_keywords)
                
                # Keyword frequency bar chart
                st.subheader("Top Keywords")
                keyword_series = pd.Series(top_keywords)
                fig = self.viz_manager.create_bar_chart(
                    keyword_series, "Keyword Frequency", "Keywords", "Frequency"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Journal Distribution")
            journal_dist = analytics.get_journal_distribution()
            
            if not journal_dist.empty:
                fig = self.viz_manager.create_pie_chart(journal_dist, "Papers by Journal")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Journal impact factors
                st.subheader("Journal Impact Factors")
                journal_if = analytics.df.groupby('journal')['impact_factor'].mean().sort_values(ascending=False)
                st.dataframe(journal_if, use_container_width=True)
        
        with tab5:
            st.subheader("Research Methodology Distribution")
            method_dist = analytics.get_methodology_distribution()
            
            if method_dist:
                method_series = pd.Series(dict(method_dist))
                fig = self.viz_manager.create_bar_chart(
                    method_series, "Research Methodologies", "Method", "Count"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Export analytics report
        st.divider()
        if st.button("📥 Export Analytics Report"):
            report = {
                'summary': stats,
                'papers': [p.to_dict() for p in st.session_state.selected_papers],
                'generated_at': datetime.now().isoformat()
            }
            
            report_json = json.dumps(report, indent=2)
            b64 = base64.b64encode(report_json.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="analytics_report.json">Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    def render_portfolio_page(self):
        """Render the research portfolio page."""
        st.header("📁 Research Portfolio")
        
        if not st.session_state.selected_papers:
            st.info("Your portfolio is empty. Add papers from the search page.")
            return
        
        st.write(f"Portfolio contains **{len(st.session_state.selected_papers)}** papers")
        
        # Portfolio actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📥 Export All (BibTeX)"):
                all_bibtex = "\n\n".join([p.to_bibtex() for p in st.session_state.selected_papers])
                st.text_area("All BibTeX Entries", all_bibtex, height=300)
        
        with col2:
            if st.button("📊 View Analytics"):
                st.switch_page("pages/analytics.py")  # Note: This would need page navigation setup
        
        with col3:
            if st.button("🗑️ Clear Portfolio"):
                st.session_state.selected_papers = []
                st.rerun()
        
        st.divider()
        
        # Display papers
        for i, paper in enumerate(st.session_state.selected_papers):
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"**{i+1}. {paper.title}**")
                    st.caption(f"{', '.join(paper.authors[:3])} | {paper.year} | {paper.journal}")
                    st.caption(f"Citations: {paper.citations} | Impact Factor: {paper.impact_factor}")
                
                with col2:
                    if st.button(f"Remove", key=f"remove_{i}"):
                        st.session_state.selected_papers.pop(i)
                        st.rerun()
                
                st.divider()
    
    def render_about_page(self):
        """Render the about page."""
        st.header("ℹ️ About")
        
        st.markdown("""
        ### Academic Literature Search & Analytics Platform
        
        **Version:** 3.1  
        **Purpose:** Advanced research tool for mechanical engineering literature
        
        #### Features:
        - 🔍 **Advanced Search**: Full-text search with filters
        - 📊 **Analytics Dashboard**: Comprehensive research analytics
        - 📁 **Portfolio Management**: Organize and export research
        - 📈 **Visualizations**: Interactive charts and graphs
        - 📚 **BibTeX Export**: Easy citation management
        
        #### Technical Stack:
        - **Framework**: Streamlit
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly (with fallback support)
        - **Network Analysis**: NetworkX (optional)
        - **Word Clouds**: WordCloud (optional)
        
        #### System Status:
        """)
        
        # Show component status
        status_data = {
            'Component': ['Plotly', 'NetworkX', 'WordCloud'],
            'Status': [
                '✅ Available' if PLOTLY_AVAILABLE else '❌ Not Available',
                '✅ Available' if NETWORKX_AVAILABLE else '❌ Not Available',
                '✅ Available' if WORDCLOUD_AVAILABLE else '❌ Not Available'
            ]
        }
        st.dataframe(pd.DataFrame(status_data), hide_index=True)
        
        st.markdown("""
        #### Usage Instructions:
        1. **Search**: Use the Literature Search page to find relevant papers
        2. **Select**: Add papers to your portfolio for analysis
        3. **Analyze**: View comprehensive analytics in the dashboard
        4. **Export**: Export citations and reports for your research
        
        #### Contact:
        For issues or suggestions, please contact the development team.
        
        ---
        *© 2024 Academic Research Tools. For educational and research purposes.*
        """)

# ================== Main Entry Point ==================

def main():
    """Main application entry point."""
    app = LiteratureSearchApp()
    app.run()

if __name__ == "__main__":
    main()
