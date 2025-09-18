#!/usr/bin/env python3
"""
Test script to verify API connections for academic databases.
Run this to check which APIs are working correctly.
"""

import sys
import time
import json

def test_import():
    """Test if required libraries are installed."""
    print("Testing library imports...")
    
    libraries = {
        'streamlit': False,
        'pandas': False,
        'numpy': False,
        'plotly': False,
        'requests': False,
        'feedparser': False,
        'arxiv': False,
        'scholarly': False
    }
    
    for lib in libraries:
        try:
            __import__(lib)
            libraries[lib] = True
            print(f"✅ {lib} installed")
        except ImportError:
            print(f"❌ {lib} NOT installed")
    
    return all(libraries.values())

def test_semantic_scholar():
    """Test Semantic Scholar API."""
    print("\n📚 Testing Semantic Scholar API...")
    try:
        import requests
        
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': 'machine learning',
            'limit': 1,
            'fields': 'title,authors,year'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                print(f"✅ Semantic Scholar API working")
                print(f"   Sample result: {data['data'][0].get('title', 'N/A')[:50]}...")
                return True
            else:
                print(f"⚠️ Semantic Scholar API returned no results")
                return False
        else:
            print(f"❌ Semantic Scholar API error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Semantic Scholar API error: {str(e)}")
        return False

def test_crossref():
    """Test CrossRef API."""
    print("\n📚 Testing CrossRef API...")
    try:
        import requests
        
        url = "https://api.crossref.org/works"
        params = {
            'query': 'machine learning',
            'rows': 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'message' in data and 'items' in data['message']:
                print(f"✅ CrossRef API working")
                items = data['message']['items']
                if items:
                    title = items[0].get('title', ['N/A'])[0]
                    print(f"   Sample result: {title[:50]}...")
                return True
            else:
                print(f"⚠️ CrossRef API returned unexpected format")
                return False
        else:
            print(f"❌ CrossRef API error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ CrossRef API error: {str(e)}")
        return False

def test_arxiv():
    """Test arXiv API."""
    print("\n📚 Testing arXiv API...")
    
    # Try Python library first
    try:
        import arxiv
        
        search = arxiv.Search(
            query="machine learning",
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = list(search.results())
        
        if results:
            print(f"✅ arXiv Python API working")
            print(f"   Sample result: {results[0].title[:50]}...")
            return True
        else:
            print(f"⚠️ arXiv Python API returned no results")
            
    except ImportError:
        print("   arXiv Python library not installed, trying HTTP API...")
    except Exception as e:
        print(f"   arXiv Python API error: {str(e)}")
    
    # Fallback to HTTP API
    try:
        import requests
        import feedparser
        
        url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': 'machine learning',
            'start': 0,
            'max_results': 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            feed = feedparser.parse(response.text)
            if feed.entries:
                print(f"✅ arXiv HTTP API working")
                print(f"   Sample result: {feed.entries[0].title[:50]}...")
                return True
            else:
                print(f"⚠️ arXiv HTTP API returned no results")
                return False
        else:
            print(f"❌ arXiv HTTP API error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ arXiv HTTP API error: {str(e)}")
        return False

def test_pubmed():
    """Test PubMed API."""
    print("\n📚 Testing PubMed API...")
    try:
        import requests
        
        # Search step
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': 'machine learning',
            'retmax': 1,
            'retmode': 'json'
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            id_list = data.get('esearchresult', {}).get('idlist', [])
            
            if id_list:
                print(f"✅ PubMed API working")
                print(f"   Found PMID: {id_list[0]}")
                return True
            else:
                print(f"⚠️ PubMed API returned no results")
                return False
        else:
            print(f"❌ PubMed API error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ PubMed API error: {str(e)}")
        return False

def test_doaj():
    """Test DOAJ API."""
    print("\n📚 Testing DOAJ API...")
    try:
        import requests
        
        url = "https://doaj.org/api/v2/search/articles"
        params = {
            'query': 'machine learning',
            'pageSize': 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            if results:
                print(f"✅ DOAJ API working")
                title = results[0].get('bibjson', {}).get('title', 'N/A')
                print(f"   Sample result: {title[:50]}...")
                return True
            else:
                print(f"⚠️ DOAJ API returned no results")
                return False
        else:
            print(f"❌ DOAJ API error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ DOAJ API error: {str(e)}")
        return False

def test_google_scholar():
    """Test Google Scholar (scholarly library)."""
    print("\n📚 Testing Google Scholar (scholarly)...")
    try:
        from scholarly import scholarly
        
        # Note: Google Scholar may rate limit or block requests
        search_query = scholarly.search_pubs('machine learning')
        
        # Get first result
        first_result = next(search_query, None)
        
        if first_result:
            print(f"✅ Google Scholar (scholarly) working")
            title = first_result.get('bib', {}).get('title', 'N/A')
            print(f"   Sample result: {title[:50]}...")
            return True
        else:
            print(f"⚠️ Google Scholar returned no results")
            return False
            
    except ImportError:
        print(f"❌ scholarly library not installed")
        return False
    except StopIteration:
        print(f"⚠️ Google Scholar returned no results")
        return False
    except Exception as e:
        print(f"⚠️ Google Scholar may be blocking requests: {str(e)}")
        print(f"   This is common - Google Scholar has strict rate limits")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Academic Database API Connection Tester")
    print("=" * 60)
    
    # Test imports first
    if not test_import():
        print("\n⚠️ Some required libraries are missing.")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test each API
    results = {
        'Semantic Scholar': test_semantic_scholar(),
        'CrossRef': test_crossref(),
        'arXiv': test_arxiv(),
        'PubMed': test_pubmed(),
        'DOAJ': test_doaj(),
        'Google Scholar': test_google_scholar()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    working_apis = sum(results.values())
    total_apis = len(results)
    
    for api, status in results.items():
        status_symbol = "✅" if status else "❌"
        print(f"{status_symbol} {api}")
    
    print(f"\n{working_apis}/{total_apis} APIs are working correctly")
    
    if working_apis < total_apis:
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Some APIs may be temporarily down")
        print("3. Google Scholar often blocks automated requests")
        print("4. Try running the test again after a few minutes")
        print("5. Install missing libraries: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
