"""
Streamlit app for SEO content analysis
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# path setup
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from streamlit_app.utils.analyzer import ContentAnalyzer
from streamlit_app.utils.visualizations import (
    plot_quality_distribution,
    plot_duplicate_heatmap,
    create_metrics_display
)

# Page configuration
st.set_page_config(
    page_title="SEO Content Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# load analyzer
@st.cache_resource
def load_analyzer():
    return ContentAnalyzer(ROOT / 'models', ROOT / 'data')

analyzer = load_analyzer()

st.title("SEO Content Quality Analyzer")
st.markdown("Analyze content quality, detect duplicates, and improve your SEO")

# Sidebar
with st.sidebar:
    st.header("Dataset Overview")
    
    stats = analyzer.get_dataset_stats()
    st.metric("Total Pages", stats['total_pages'])
    st.metric("Duplicate Pairs", stats['duplicate_pairs'])
    st.metric("Thin Content", f"{stats['thin_content_pct']:.1f}%")
    
    st.divider()
    
    st.header("Settings")
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Minimum similarity score to show related content"
    )
    
    st.divider()
    
    st.markdown("""
    ### About
    Analyzes web content for quality scoring, duplicate detection, and readability metrics.
    
    Built with Streamlit
    """)

# Main tabs
tab1, tab2, tab3 = st.tabs(["Analyze URL", "Dataset Insights", "Documentation"])

# Tab 1: URL analysis
with tab1:
    st.header("Analyze a URL")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com/page",
            label_visibility="collapsed"
        )
    with col2:
        st.write("")
        st.write("")
        analyze_button = st.button("Analyze", type="primary", use_container_width=True)
    
    if analyze_button and url_input:
        with st.spinner("Analyzing..."):
            result = analyzer.analyze_url(url_input, similarity_threshold)
        
        if 'error' in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success("Analysis complete!")
            
            # show metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Quality Score", f"{result['quality_score']:.1f}%")
            with col2:
                st.metric("Classification", result['quality_label'])
            with col3:
                st.metric("Word Count", result['word_count'])
            with col4:
                st.metric("Readability", f"{result['readability']:.1f}")
            
            st.divider()
            
            # Content metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Content Metrics")
                metrics_data = {
                    "Metric": ["Sentence Count", "Paragraph Count", "Avg Words/Sentence", "Unique Words"],
                    "Value": [
                        result['sentence_count'],
                        result['paragraph_count'],
                        f"{result['avg_words_per_sentence']:.1f}",
                        result['unique_words']
                    ]
                }
                st.table(pd.DataFrame(metrics_data))
            
            with col2:
                st.subheader("HTML Elements")
                html_data = {
                    "Element": ["Images", "Links", "Headings"],
                    "Count": [
                        result['image_count'],
                        result['link_count'],
                        result['heading_count']
                    ]
                }
                st.table(pd.DataFrame(html_data))
            
            # Similar content
            if result['similar_content']:
                st.subheader("Similar Content Found")
                for item in result['similar_content']:
                    with st.expander(f"Similarity: {item['similarity']:.1%} - {item['url']}"):
                        st.write(f"**Word Count:** {item['word_count']}")
                        st.write(f"**Quality:** {item['quality_label']}")
            else:
                st.info("No similar content found in the dataset")

# Tab 2: Dataset Insights
with tab2:
    st.header("Dataset Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_quality_distribution(analyzer.df_features), use_container_width=True)
    with col2:
        fig = plot_duplicate_heatmap(analyzer.df_duplicates)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No duplicate pairs found")
    
    st.divider()
    
    # Top quality content
    st.subheader("Top Quality Content")
    top_content = analyzer.df_features.nlargest(5, 'quality_score')[['url', 'quality_score', 'quality_label', 'word_count']]
    
    for idx, row in top_content.iterrows():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"**{row['url']}**")
        with col2:
            st.write(row['quality_label'])
        with col3:
            st.metric("Score", f"{row['quality_score']:.1f}%")
    
    st.divider()
    
    # Thin content
    st.subheader("Thin Content Pages")
    thin_content = analyzer.df_features[analyzer.df_features['word_count'] < 300].nlargest(5, 'word_count')[['url', 'word_count', 'quality_score']]
    
    if not thin_content.empty:
        for idx, row in thin_content.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{row['url']}**")
            with col2:
                st.metric("Words", row['word_count'])
    else:
        st.success("No thin content found!")

# Tab 3: Documentation
with tab3:
    st.header("Documentation")
    
    st.markdown("""
    ## Features
    
    - **Quality Scoring**: ML-powered analysis of content quality
    - **Duplicate Detection**: Find similar/duplicate pages
    - **SEO Metrics**: Word count, readability, content structure
    - **Content Analysis**: Headings, links, images, keywords
    
    ## Quality Classification
    
    Content is classified into three categories:
    
    - **High Quality**: Comprehensive content with good structure
    - **Medium Quality**: Adequate content with room for improvement
    - **Low Quality**: Thin content that needs work
    
    ## Metrics Explained
    
    ### Quality Score (0-100)
    Machine learning model trained on content features including:
    - Content length and depth
    - Readability and structure
    - HTML elements and formatting
    - Keyword usage and relevance
    
    ### Readability Score
    Flesch Reading Ease score (0-100):
    - 90-100: Very easy to read
    - 60-70: Plain English
    - 0-30: Very difficult to read
    
    ### Duplicate Detection
    Uses cosine similarity with TF-IDF vectors to compare content text similarity.
    Adjustable threshold from 0.0 (completely different) to 1.0 (identical).
    """)
