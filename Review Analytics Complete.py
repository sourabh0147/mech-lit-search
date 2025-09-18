"""
Integrated Mechanical Engineering Literature Search and Analytics Application
Version 3.0 - Enhanced with Comprehensive Analytics Features

This application provides a sophisticated interface for searching academic literature
with integrated analytics capabilities for research paper analysis and visualization.

Author: Sourabh
Date: September 2025
License: Academic Use
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter, defaultdict
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
import json
import re
from dataclasses import dataclass, field
import logging
from enum import Enum

# Configure logging for academic research standards
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================== Data Models ==================

@dataclass
class Paper:
    """Representation of an academic paper with comprehensive metadata."""
    
    title: str
    authors: List[str]
    year: int
    abstract: str
    keywords: List[str]
    journal: str
    doi: str
    citations: int
    references: List[str] = field(default_factory=list)
    venue_type: str = "journal"
    impact_factor: float = 0.0
    field_of_study: List[str] = field(default_factory=list)
    methodology: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert paper object to dictionary for serialization."""
        return {
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'abstract': self.abstract,
            'keywords': self.keywords,
            'journal': self.journal,
            'doi': self.doi,
            'citations': self.citations,
            'references': self.references,
            'venue_type': self.venue_type,
            'impact_factor': self.impact_factor,
            'field_of_study': self.field_of_study,
            'methodology': self.methodology
        }


class SearchCriteria(Enum):
    """Enumeration of available search criteria for academic literature."""
    
    TITLE = "title"
    AUTHOR = "author"
    KEYWORD = "keyword"
    YEAR_RANGE = "year_range"
    JOURNAL = "journal"
    MIN_CITATIONS = "min_citations"
    METHODOLOGY = "methodology"
    FIELD = "field_of_study"


# ================== Search Engine Module ==================

class LiteratureSearchEngine:
    """
    Core search engine for academic literature retrieval.
    Implements advanced search algorithms with relevance scoring.
    """
    
    def __init__(self):
        self.papers_database = self._initialize_sample_database()
        self.index = self._build_search_index()
    
    def _initialize_sample_database(self) -> List[Paper]:
        """Initialize sample database with academic papers for demonstration."""
        return [
            Paper(
                title="Deep Learning Applications in Mechanical Fault Detection: A Comprehensive Survey",
                authors=["Zhang, W.", "Li, X.", "Wang, S."],
                year=2024,
                abstract="This paper presents a comprehensive survey of deep learning techniques applied to mechanical fault detection...",
                keywords=["deep learning", "fault detection", "mechanical systems", "neural networks"],
                journal="Mechanical Systems and Signal Processing",
                doi="10.1016/j.ymssp.2024.110234",
                citations=45,
                references=["DOI:10.1016/j.ymssp.2023.109876", "DOI:10.1109/TII.2023.3245678"],
                impact_factor=6.8,
                field_of_study=["Mechanical Engineering", "Artificial Intelligence"],
                methodology=["Survey", "Systematic Review"]
            ),
            Paper(
                title="Advances in Composite Materials for Aerospace Applications",
                authors=["Johnson, M.A.", "Smith, R.T.", "Chen, L."],
                year=2023,
                abstract="Recent developments in composite materials have revolutionized aerospace engineering...",
                keywords=["composite materials", "aerospace", "carbon fiber", "manufacturing"],
                journal="Composites Part A: Applied Science and Manufacturing",
                doi="10.1016/j.compositesa.2023.107234",
                citations=78,
                references=["DOI:10.1016/j.compositesa.2022.106789"],
                impact_factor=7.2,
                field_of_study=["Materials Science", "Aerospace Engineering"],
                methodology=["Experimental", "Computational Modeling"]
            ),
            Paper(
                title="Optimization of Wind Turbine Blade Design Using Genetic Algorithms",
                authors=["Anderson, K.", "Liu, Y.", "Martinez, J."],
                year=2024,
                abstract="This study presents a novel approach to wind turbine blade optimization using genetic algorithms...",
                keywords=["wind energy", "optimization", "genetic algorithms", "renewable energy"],
                journal="Renewable Energy",
                doi="10.1016/j.renene.2024.119234",
                citations=23,
                references=["DOI:10.1016/j.renene.2023.118456"],
                impact_factor=8.1,
                field_of_study=["Renewable Energy", "Mechanical Engineering"],
                methodology=["Optimization", "Simulation"]
            ),
            Paper(
                title="Machine Learning for Predictive Maintenance in Manufacturing Systems",
                authors=["Wang, H.", "Thompson, D.", "Kumar, A."],
                year=2023,
                abstract="Predictive maintenance using machine learning has become crucial for modern manufacturing...",
                keywords=["predictive maintenance", "machine learning", "manufacturing", "Industry 4.0"],
                journal="Journal of Manufacturing Systems",
                doi="10.1016/j.jmsy.2023.05.012",
                citations=92,
                references=["DOI:10.1016/j.jmsy.2022.04.008"],
                impact_factor=8.6,
                field_of_study=["Manufacturing Engineering", "Data Science"],
                methodology=["Machine Learning", "Case Study"]
            ),
            Paper(
                title="Thermal Management in Electric Vehicle Battery Systems: A Review",
                authors=["Park, J.", "Wilson, E.", "Zhang, Q."],
                year=2024,
                abstract="Effective thermal management is critical for electric vehicle battery performance and safety...",
                keywords=["electric vehicles", "battery thermal management", "cooling systems", "heat transfer"],
                journal="Applied Thermal Engineering",
                doi="10.1016/j.applthermaleng.2024.122345",
                citations=56,
                references=["DOI:10.1016/j.applthermaleng.2023.121234"],
                impact_factor=5.4,
                field_of_study=["Thermal Engineering", "Automotive Engineering"],
                methodology=["Review", "Analytical Modeling"]
            )
        ]
    
    def _build_search_index(self) -> Dict[str, List[int]]:
        """Build inverted index for efficient text search."""
        index = defaultdict(list)
        
        for idx, paper in enumerate(self.papers_database):
            # Index title words
            for word in paper.title.lower().split():
                index[word].append(idx)
            
            # Index abstract words
            for word in paper.abstract.lower().split():
                index[word].append(idx)
            
            # Index keywords
            for keyword in paper.keywords:
                for word in keyword.lower().split():
                    index[word].append(idx)
            
            # Index authors
            for author in paper.authors:
                index[author.lower()].append(idx)
        
        return dict(index)
    
    def search(self, query: str, criteria: Dict[str, Any]) -> List[Paper]:
        """
        Execute search based on query and criteria.
        
        Parameters:
        -----------
        query : str
            Search query string
        criteria : Dict[str, Any]
            Additional search criteria
        
        Returns:
        --------
        List[Paper]
            Filtered and ranked search results
        """
        results = []
        query_words = query.lower().split()
        
        # Find papers matching query
        paper_scores = defaultdict(float)
        for word in query_words:
            if word in self.index:
                for paper_idx in self.index[word]:
                    paper_scores[paper_idx] += 1.0
        
        # Apply additional filters
        for paper_idx, score in paper_scores.items():
            paper = self.papers_database[paper_idx]
            
            # Year filter
            if 'year_min' in criteria and paper.year < criteria['year_min']:
                continue
            if 'year_max' in criteria and paper.year > criteria['year_max']:
                continue
            
            # Citation filter
            if 'min_citations' in criteria and paper.citations < criteria['min_citations']:
                continue
            
            # Journal filter
            if 'journal' in criteria and criteria['journal']:
                if criteria['journal'].lower() not in paper.journal.lower():
                    continue
            
            # Field filter
            if 'field' in criteria and criteria['field']:
                if not any(field in paper.field_of_study for field in criteria['field']):
                    continue
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(paper, query, score)
            results.append((paper, relevance_score))
        
        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [paper for paper, _ in results]
    
    def _calculate_relevance_score(self, paper: Paper, query: str, base_score: float) -> float:
        """Calculate comprehensive relevance score for ranking."""
        score = base_score
        
        # Boost for title match
        if query.lower() in paper.title.lower():
            score *= 2.0
        
        # Boost for recent papers
        current_year = datetime.now().year
        recency_factor = 1.0 + (paper.year - 2020) / 10.0
        score *= recency_factor
        
        # Boost for citations
        citation_factor = 1.0 + np.log1p(paper.citations) / 10.0
        score *= citation_factor
        
        # Boost for impact factor
        impact_factor = 1.0 + paper.impact_factor / 10.0
        score *= impact_factor
        
        return score


# ================== Analytics Module ==================

class ResearchAnalytics:
    """
    Advanced analytics module for research paper analysis.
    Provides statistical insights, trends, and visualizations.
    """
    
    def __init__(self, papers: List[Paper]):
        self.papers = papers
        self.df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert papers to pandas DataFrame for analysis."""
        data = []
        for paper in self.papers:
            data.append({
                'title': paper.title,
                'authors': ', '.join(paper.authors),
                'num_authors': len(paper.authors),
                'year': paper.year,
                'journal': paper.journal,
                'citations': paper.citations,
                'impact_factor': paper.impact_factor,
                'num_keywords': len(paper.keywords),
                'keywords': ', '.join(paper.keywords),
                'field': ', '.join(paper.field_of_study),
                'methodology': ', '.join(paper.methodology)
            })
        return pd.DataFrame(data)
    
    def generate_temporal_analysis(self) -> Dict[str, Any]:
        """Analyze temporal trends in the research corpus."""
        analysis = {}
        
        # Publications per year
        year_counts = self.df['year'].value_counts().sort_index()
        analysis['publications_per_year'] = year_counts.to_dict()
        
        # Citation trends
        citation_by_year = self.df.groupby('year')['citations'].agg(['mean', 'sum', 'max'])
        analysis['citation_trends'] = citation_by_year.to_dict()
        
        # Growth rate
        if len(year_counts) > 1:
            growth_rate = (year_counts.iloc[-1] - year_counts.iloc[0]) / len(year_counts)
            analysis['average_growth_rate'] = growth_rate
        
        return analysis
    
    def generate_author_analysis(self) -> Dict[str, Any]:
        """Analyze author collaboration patterns and productivity."""
        analysis = {}
        
        # Author frequency
        all_authors = []
        for authors in self.df['authors']:
            all_authors.extend(authors.split(', '))
        
        author_counts = Counter(all_authors)
        analysis['most_productive_authors'] = dict(author_counts.most_common(10))
        
        # Collaboration metrics
        analysis['avg_authors_per_paper'] = self.df['num_authors'].mean()
        analysis['collaboration_index'] = (self.df['num_authors'] > 1).mean()
        
        # Author network
        analysis['collaboration_network'] = self._build_collaboration_network()
        
        return analysis
    
    def _build_collaboration_network(self) -> Dict[str, List[str]]:
        """Build author collaboration network."""
        network = defaultdict(set)
        
        for authors in self.df['authors']:
            author_list = authors.split(', ')
            for i, author1 in enumerate(author_list):
                for author2 in author_list[i+1:]:
                    network[author1].add(author2)
                    network[author2].add(author1)
        
        return {k: list(v) for k, v in network.items()}
    
    def generate_keyword_analysis(self) -> Dict[str, Any]:
        """Analyze keyword trends and co-occurrences."""
        analysis = {}
        
        # Keyword frequency
        all_keywords = []
        for keywords in self.df['keywords']:
            all_keywords.extend(keywords.split(', '))
        
        keyword_counts = Counter(all_keywords)
        analysis['top_keywords'] = dict(keyword_counts.most_common(15))
        
        # Keyword co-occurrence
        cooccurrence = defaultdict(int)
        for keywords in self.df['keywords']:
            keyword_list = keywords.split(', ')
            for i, kw1 in enumerate(keyword_list):
                for kw2 in keyword_list[i+1:]:
                    pair = tuple(sorted([kw1, kw2]))
                    cooccurrence[pair] += 1
        
        analysis['keyword_cooccurrence'] = dict(sorted(
            cooccurrence.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])
        
        return analysis
    
    def generate_impact_analysis(self) -> Dict[str, Any]:
        """Analyze research impact metrics."""
        analysis = {}
        
        # Citation statistics
        analysis['total_citations'] = self.df['citations'].sum()
        analysis['mean_citations'] = self.df['citations'].mean()
        analysis['median_citations'] = self.df['citations'].median()
        analysis['h_index'] = self._calculate_h_index()
        
        # Impact factor analysis
        analysis['mean_impact_factor'] = self.df['impact_factor'].mean()
        analysis['high_impact_papers'] = len(self.df[self.df['impact_factor'] > 5])
        
        # Field impact
        field_impact = self.df.groupby('field')['citations'].mean()
        analysis['field_impact'] = field_impact.to_dict()
        
        return analysis
    
    def _calculate_h_index(self) -> int:
        """Calculate h-index for the paper collection."""
        citations = sorted(self.df['citations'].values, reverse=True)
        h_index = 0
        for i, c in enumerate(citations, 1):
            if c >= i:
                h_index = i
            else:
                break
        return h_index
    
    def generate_methodology_analysis(self) -> Dict[str, Any]:
        """Analyze research methodologies used."""
        analysis = {}
        
        # Methodology frequency
        all_methods = []
        for methods in self.df['methodology']:
            all_methods.extend(methods.split(', '))
        
        method_counts = Counter(all_methods)
        analysis['methodology_distribution'] = dict(method_counts)
        
        # Methodology trends over time
        method_trends = defaultdict(lambda: defaultdict(int))
        for _, row in self.df.iterrows():
            for method in row['methodology'].split(', '):
                method_trends[method][row['year']] += 1
        
        analysis['methodology_trends'] = dict(method_trends)
        
        return analysis
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        report = {
            'summary_statistics': {
                'total_papers': len(self.papers),
                'date_range': f"{self.df['year'].min()} - {self.df['year'].max()}",
                'unique_journals': self.df['journal'].nunique(),
                'unique_authors': len(set(sum([a.split(', ') for a in self.df['authors']], [])))
            },
            'temporal_analysis': self.generate_temporal_analysis(),
            'author_analysis': self.generate_author_analysis(),
            'keyword_analysis': self.generate_keyword_analysis(),
            'impact_analysis': self.generate_impact_analysis(),
            'methodology_analysis': self.generate_methodology_analysis()
        }
        
        return report


# ================== Visualization Module ==================

class VisualizationEngine:
    """
    Sophisticated visualization engine for research analytics.
    Generates publication-quality charts and graphs.
    """
    
    @staticmethod
    def create_temporal_trend_chart(data: Dict[int, int]) -> go.Figure:
        """Create temporal trend visualization."""
        years = list(data.keys())
        counts = list(data.values())
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years,
            y=counts,
            mode='lines+markers',
            name='Publications',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Temporal Distribution of Publications',
            xaxis_title='Year',
            yaxis_title='Number of Publications',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_keyword_network(cooccurrence: Dict[Tuple[str, str], int]) -> go.Figure:
        """Create keyword co-occurrence network visualization."""
        G = nx.Graph()
        
        for (kw1, kw2), weight in cooccurrence.items():
            G.add_edge(kw1, kw2, weight=weight)
        
        pos = nx.spring_layout(G)
        
        edge_trace = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=edge[2]['weight'], color='#888'),
                hoverinfo='none'
            ))
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            text=[node for node in G.nodes()],
            mode='markers+text',
            textposition='top center',
            marker=dict(
                size=15,
                color='#2E86AB',
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            title='Keyword Co-occurrence Network',
            showlegend=False,
            template='plotly_white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    @staticmethod
    def create_citation_distribution(citations: List[int]) -> go.Figure:
        """Create citation distribution histogram."""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=citations,
            nbinsx=20,
            name='Citations',
            marker_color='#2E86AB'
        ))
        
        fig.update_layout(
            title='Citation Distribution Analysis',
            xaxis_title='Number of Citations',
            yaxis_title='Frequency',
            template='plotly_white',
            bargap=0.1
        )
        
        return fig
    
    @staticmethod
    def create_author_productivity_chart(author_counts: Dict[str, int]) -> go.Figure:
        """Create author productivity bar chart."""
        authors = list(author_counts.keys())[:10]
        counts = list(author_counts.values())[:10]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=counts,
            y=authors,
            orientation='h',
            marker_color='#2E86AB'
        ))
        
        fig.update_layout(
            title='Most Productive Authors',
            xaxis_title='Number of Publications',
            yaxis_title='Author',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_impact_matrix(df: pd.DataFrame) -> go.Figure:
        """Create impact factor vs citations scatter plot."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['impact_factor'],
            y=df['citations'],
            mode='markers',
            text=df['title'],
            marker=dict(
                size=10,
                color=df['year'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Year')
            )
        ))
        
        fig.update_layout(
            title='Research Impact Matrix',
            xaxis_title='Journal Impact Factor',
            yaxis_title='Citations',
            template='plotly_white',
            hovermode='closest'
        )
        
        return fig


# ================== Main Application ==================

class IntegratedLiteratureApp:
    """
    Main application class integrating search and analytics functionalities.
    Provides comprehensive interface for academic literature exploration.
    """
    
    def __init__(self):
        st.set_page_config(
            page_title="Academic Literature Search & Analytics",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.search_engine = LiteratureSearchEngine()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'selected_papers' not in st.session_state:
            st.session_state.selected_papers = []
        if 'analytics_report' not in st.session_state:
            st.session_state.analytics_report = None
    
    def run(self):
        """Execute main application workflow."""
        self.render_header()
        
        # Sidebar navigation
        with st.sidebar:
            st.title("Navigation")
            mode = st.radio(
                "Select Mode",
                ["Literature Search", "Analytics Dashboard", "Research Portfolio"]
            )
        
        if mode == "Literature Search":
            self.render_search_interface()
        elif mode == "Analytics Dashboard":
            self.render_analytics_dashboard()
        else:
            self.render_research_portfolio()
    
    def render_header(self):
        """Render application header with academic styling."""
        st.title("🎓 Integrated Academic Literature Search & Analytics Platform")
        st.markdown("""
        <style>
        .main-header {
            padding: 1rem;
            background: linear-gradient(90deg, #2E86AB 0%, #1E5A7D 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Advanced Research Tool for Mechanical Engineering Literature**  
        *Version 3.0 - Enhanced with Comprehensive Analytics Suite*
        """)
        st.divider()
    
    def render_search_interface(self):
        """Render literature search interface."""
        st.header("📖 Literature Search Module")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input(
                "Enter Search Query",
                placeholder="e.g., deep learning mechanical fault detection"
            )
        
        with col2:
            st.markdown("### Advanced Filters")
            
        # Advanced search criteria
        with st.expander("Advanced Search Criteria", expanded=False):
            col_a, col_b = st.columns(2)
            
            with col_a:
                year_range = st.slider(
                    "Publication Year Range",
                    2000, 2024, (2020, 2024)
                )
                
                min_citations = st.number_input(
                    "Minimum Citations",
                    min_value=0,
                    value=0
                )
            
            with col_b:
                journal_filter = st.text_input(
                    "Journal Name (optional)",
                    placeholder="e.g., Mechanical Systems"
                )
                
                field_filter = st.multiselect(
                    "Field of Study",
                    ["Mechanical Engineering", "Materials Science", 
                     "Artificial Intelligence", "Renewable Energy",
                     "Manufacturing Engineering", "Aerospace Engineering"]
                )
        
        # Search execution
        if st.button("🔍 Execute Search", type="primary"):
            if search_query:
                criteria = {
                    'year_min': year_range[0],
                    'year_max': year_range[1],
                    'min_citations': min_citations,
                    'journal': journal_filter,
                    'field': field_filter
                }
                
                with st.spinner("Searching academic databases..."):
                    results = self.search_engine.search(search_query, criteria)
                    st.session_state.search_results = results
                
                st.success(f"Found {len(results)} relevant papers")
        
        # Display search results
        if st.session_state.search_results:
            st.markdown("### Search Results")
            
            for idx, paper in enumerate(st.session_state.search_results):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{paper.title}**")
                        st.caption(f"Authors: {', '.join(paper.authors)}")
                        st.caption(f"Journal: {paper.journal} ({paper.year})")
                    
                    with col2:
                        st.metric("Citations", paper.citations)
                    
                    with col3:
                        if st.button(f"Select", key=f"select_{idx}"):
                            if paper not in st.session_state.selected_papers:
                                st.session_state.selected_papers.append(paper)
                                st.success("Added to portfolio")
                    
                    with st.expander("View Abstract & Details"):
                        st.write(paper.abstract)
                        st.markdown(f"**Keywords:** {', '.join(paper.keywords)}")
                        st.markdown(f"**DOI:** {paper.doi}")
                        st.markdown(f"**Impact Factor:** {paper.impact_factor}")
                    
                    st.divider()
    
    def render_analytics_dashboard(self):
        """Render comprehensive analytics dashboard."""
        st.header("📊 Research Analytics Dashboard")
        
        if not st.session_state.selected_papers:
            st.warning("Please select papers from the search interface first.")
            return
        
        # Initialize analytics
        analytics = ResearchAnalytics(st.session_state.selected_papers)
        viz_engine = VisualizationEngine()
        
        # Generate comprehensive report
        report = analytics.generate_comprehensive_report()
        st.session_state.analytics_report = report
        
        # Summary metrics
        st.markdown("### Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Papers", report['summary_statistics']['total_papers'])
        with col2:
            st.metric("Date Range", report['summary_statistics']['date_range'])
        with col3:
            st.metric("Unique Authors", report['summary_statistics']['unique_authors'])
        with col4:
            st.metric("H-Index", report['impact_analysis']['h_index'])
        
        st.divider()
        
        # Visualization tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Temporal Analysis", 
            "Author Networks", 
            "Keyword Analysis",
            "Impact Metrics",
            "Methodology Trends"
        ])
        
        with tab1:
            st.subheader("Temporal Distribution of Research")
            if report['temporal_analysis']['publications_per_year']:
                fig = viz_engine.create_temporal_trend_chart(
                    report['temporal_analysis']['publications_per_year']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Citation trends
            st.subheader("Citation Trends Over Time")
            citation_trends = report['temporal_analysis']['citation_trends']
            if citation_trends:
                df_trends = pd.DataFrame(citation_trends)
                st.line_chart(df_trends)
        
        with tab2:
            st.subheader("Author Collaboration Analysis")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Most Productive Authors**")
                if report['author_analysis']['most_productive_authors']:
                    fig = viz_engine.create_author_productivity_chart(
                        report['author_analysis']['most_productive_authors']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Collaboration Metrics**")
                st.metric(
                    "Average Authors per Paper",
                    f"{report['author_analysis']['avg_authors_per_paper']:.2f}"
                )
                st.metric(
                    "Collaboration Index",
                    f"{report['author_analysis']['collaboration_index']:.2%}"
                )
        
        with tab3:
            st.subheader("Keyword Analysis")
            
            # Top keywords
            st.markdown("**Most Frequent Keywords**")
            keywords = report['keyword_analysis']['top_keywords']
            if keywords:
                df_keywords = pd.DataFrame(
                    list(keywords.items()),
                    columns=['Keyword', 'Frequency']
                )
                st.bar_chart(df_keywords.set_index('Keyword'))
            
            # Keyword co-occurrence network
            if report['keyword_analysis']['keyword_cooccurrence']:
                st.markdown("**Keyword Co-occurrence Network**")
                fig = viz_engine.create_keyword_network(
                    report['keyword_analysis']['keyword_cooccurrence']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Research Impact Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Citation Statistics**")
                st.metric("Total Citations", report['impact_analysis']['total_citations'])
                st.metric("Mean Citations", f"{report['impact_analysis']['mean_citations']:.2f}")
                st.metric("Median Citations", report['impact_analysis']['median_citations'])
            
            with col2:
                st.markdown("**Impact Factor Analysis**")
                st.metric(
                    "Mean Impact Factor",
                    f"{report['impact_analysis']['mean_impact_factor']:.2f}"
                )
                st.metric(
                    "High Impact Papers (IF > 5)",
                    report['impact_analysis']['high_impact_papers']
                )
            
            # Citation distribution
            st.markdown("**Citation Distribution**")
            citations = [p.citations for p in st.session_state.selected_papers]
            fig = viz_engine.create_citation_distribution(citations)
            st.plotly_chart(fig, use_container_width=True)
            
            # Impact matrix
            st.markdown("**Impact Matrix**")
            fig = viz_engine.create_impact_matrix(analytics.df)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.subheader("Research Methodology Analysis")
            
            # Methodology distribution
            st.markdown("**Methodology Distribution**")
            methods = report['methodology_analysis']['methodology_distribution']
            if methods:
                df_methods = pd.DataFrame(
                    list(methods.items()),
                    columns=['Methodology', 'Count']
                )
                st.bar_chart(df_methods.set_index('Methodology'))
            
            # Methodology trends
            st.markdown("**Methodology Trends Over Time**")
            method_trends = report['methodology_analysis']['methodology_trends']
            if method_trends:
                for method, yearly_data in method_trends.items():
                    st.markdown(f"**{method}**")
                    df_trend = pd.DataFrame(
                        list(yearly_data.items()),
                        columns=['Year', 'Count']
                    )
                    st.line_chart(df_trend.set_index('Year'))
    
    def render_research_portfolio(self):
        """Render research portfolio management interface."""
        st.header("📁 Research Portfolio Management")
        
        if not st.session_state.selected_papers:
            st.info("Your research portfolio is empty. Add papers from the search interface.")
            return
        
        st.markdown(f"### Portfolio Contains {len(st.session_state.selected_papers)} Papers")
        
        # Portfolio actions
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("📥 Export Portfolio (JSON)"):
                portfolio_data = [p.to_dict() for p in st.session_state.selected_papers]
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(portfolio_data, indent=2),
                    file_name=f"research_portfolio_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Generate Report"):
                if st.session_state.analytics_report:
                    st.success("Analytics report available in dashboard")
        
        with col3:
            if st.button("🗑️ Clear Portfolio"):
                st.session_state.selected_papers = []
                st.rerun()
        
        st.divider()
        
        # Display portfolio papers
        for idx, paper in enumerate(st.session_state.selected_papers):
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"**{idx + 1}. {paper.title}**")
                    st.caption(f"Authors: {', '.join(paper.authors)} | Year: {paper.year}")
                    st.caption(f"Journal: {paper.journal} | Citations: {paper.citations}")
                    st.caption(f"Keywords: {', '.join(paper.keywords[:5])}")
                
                with col2:
                    if st.button(f"Remove", key=f"remove_{idx}"):
                        st.session_state.selected_papers.pop(idx)
                        st.rerun()
                
                st.divider()
        
        # Portfolio statistics
        st.markdown("### Portfolio Statistics")
        
        if len(st.session_state.selected_papers) > 0:
            analytics = ResearchAnalytics(st.session_state.selected_papers)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_citations = sum(p.citations for p in st.session_state.selected_papers)
                st.metric("Total Citations", total_citations)
            
            with col2:
                avg_impact = np.mean([p.impact_factor for p in st.session_state.selected_papers])
                st.metric("Average Impact Factor", f"{avg_impact:.2f}")
            
            with col3:
                year_range = f"{min(p.year for p in st.session_state.selected_papers)}-{max(p.year for p in st.session_state.selected_papers)}"
                st.metric("Year Range", year_range)


# ================== Application Entry Point ==================

def main():
    """Main entry point for the application."""
    app = IntegratedLiteratureApp()
    app.run()


if __name__ == "__main__":
    main()
