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
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .quality-high { 
        color: #fff;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .quality-medium { 
        color: #fff;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .quality-low { 
        color: #fff;
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Content container */
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 12px;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_analyzer():
    return ContentAnalyzer(ROOT / 'models', ROOT / 'data')

analyzer = load_analyzer()

# Hero section with icon
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div style="font-size: 4rem; margin-bottom: 1rem;">🎯</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">SEO Analyzer Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Content Analysis • Quality Scoring • Duplicate Detection</div>', unsafe_allow_html=True)

# Sidebar with modern design
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="margin: 0;">📊 Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    stats = analyzer.get_dataset_stats()
    
    st.markdown("### 📈 Dataset Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pages", f"{stats['total_pages']:,}")
    with col2:
        st.metric("Duplicates", stats['duplicate_pairs'])
    
    st.metric("Thin Content", f"{stats['thin_content_pct']:.1f}%")
    
    st.divider()
    
    st.markdown("### ⚙️ Configuration")
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
    <div style="text-align: center; padding: 1rem 0;">
        <h3>ℹ️ About</h3>
        <p style="font-size: 0.9rem; line-height: 1.6;">
        AI-powered content analysis with quality scoring, duplicate detection, and SEO insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main tabs with icons
tab1, tab2, tab3 = st.tabs(["🔍 Analyze URL", "📊 Dataset Insights", "📚 Documentation"])

# Tab 1: URL analysis
with tab1:
    st.markdown("### 🌐 Enter URL for Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url_input = st.text_input(
            "Enter URL to analyze",
            placeholder="https://example.com/article",
            help="Enter a valid URL to analyze its content quality"
        )
    
    with col2:
        st.write("")
        st.write("")
        analyze_button = st.button("🚀 Analyze Now", type="primary", use_container_width=True)
    
    if analyze_button and url_input:
        with st.spinner("🔄 Analyzing content... Please wait"):
            result = analyzer.analyze_url(url_input, similarity_threshold)
        
        if 'error' in result:
            st.error(f"❌ {result['error']}")
        else:
            st.success("✅ Analysis Complete!")
            
            # show metrics with modern cards
            st.markdown("### 📈 Results")
            col1, col2, col3, col4 = st.columns(4)
            
            quality_class = f"quality-{result['quality_label'].lower()}"
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #666;">Quality Score</div>
                    <div class="{quality_class}" style="font-size: 1.8rem;">{result['quality_label']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #666;">Word Count</div>
                    <div style="font-size: 1.8rem;">{result['word_count']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #666;">Readability</div>
                    <div style="font-size: 1.8rem;">{result['flesch_reading_ease']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                thin_label = "Yes" if result['is_thin_content'] else "No"
                thin_color = "#dc3545" if result['is_thin_content'] else "#28a745"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #666;">Thin Content</div>
                    <div style="font-size: 1.8rem; color: {thin_color};">{thin_label}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Content details
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Content Details")
                if result.get('title'):
                    st.markdown(f"**Title:** {result['title']}")
                st.markdown(f"**URL:** {result['url']}")
                st.markdown(f"**Sentences:** {result['sentence_count']}")
                
                # Recommendations
                st.subheader("Recommendations")
                recommendations = analyzer.get_recommendations(result)
                for rec in recommendations:
                    st.info(rec)
            
            with col2:
                st.subheader("Similar Content")
                similar_pages = result.get('similar_pages', [])
                
                if similar_pages:
                    for i, page in enumerate(similar_pages, 1):
                        with st.expander(f"Match {i} ({page['similarity']*100:.1f}% similar)"):
                            st.markdown(f"**Title:** {page.get('title', 'N/A')}")
                            st.markdown(f"**URL:** {page['url']}")
                else:
                    st.info("No similar content found in the dataset.")

# Tab 2: Dataset Insights
with tab2:
    st.markdown("### 📊 Content Analytics Dashboard")
    
    # Quality distribution
    st.subheader("Content Quality Distribution")
    quality_dist = analyzer.get_quality_distribution()
    fig_quality = plot_quality_distribution(quality_dist)
    st.plotly_chart(fig_quality, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Statistics")
        stats_df = analyzer.get_feature_statistics()
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.subheader("Duplicate Pairs")
        duplicates_df = analyzer.get_duplicates()
        if len(duplicates_df) > 0:
            st.dataframe(
                duplicates_df.head(10),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No duplicate pairs found at threshold ≥ 0.80")
    
    # Top/Bottom content
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Highest Quality Content")
        top_content = analyzer.get_top_content(n=5)
        for i, item in enumerate(top_content, 1):
            title = str(item.get('title', '')) if pd.notna(item.get('title')) else 'No title'
            title_short = title[:50] if len(title) > 50 else title
            with st.expander(f"{i}. {title_short}..."):
                st.markdown(f"**Words:** {item['word_count']}")
                st.markdown(f"**Readability:** {item['flesch_reading_ease']:.1f}")
                st.markdown(f"**URL:** {item['url']}")
    
    with col2:
        st.subheader("Thin Content Pages")
        thin_content = analyzer.get_thin_content(n=5)
        for i, item in enumerate(thin_content, 1):
            title = str(item.get('title', '')) if pd.notna(item.get('title')) else 'No title'
            title_short = title[:50] if len(title) > 50 else title
            with st.expander(f"{i}. {title_short}..."):
                st.markdown(f"**Words:** {item['word_count']}")
                st.markdown(f"**Readability:** {item['flesch_reading_ease']:.1f}")
                st.markdown(f"**URL:** {item['url']}")

# Tab 3: Documentation
with tab3:
    st.markdown("### 📚 How It Works")
    
    st.markdown("""
    ## How It Works
    
    This tool uses machine learning to analyze web content and provide actionable insights:
    
    ### 1. Content Quality Scoring
    
    Pages are classified into three categories:
    
    - **High Quality**: >1500 words with optimal readability (Flesch score 50-70)
    - **Medium Quality**: Moderate length and readability
    - **Low Quality**: <500 words or poor readability (<30)
    
    ### 2. Duplicate Detection
    
    Uses TF-IDF vectorization and cosine similarity to find near-duplicate content:
    - Threshold: 0.80 (80% similarity)
    - Method: Text-based comparison of main content
    
    ### 3. Readability Analysis
    
    Flesch Reading Ease score interpretation:
    - **90-100**: Very easy (5th grade)
    - **60-70**: Standard (8th-9th grade)
    - **30-50**: Difficult (college level)
    - **0-30**: Very difficult (professional)
    
    ### 4. SEO Recommendations
    
    The tool provides specific recommendations based on:
    - Word count targets
    - Readability optimization
    - Content uniqueness
    
    ---
    
    ## Technical Details
    
    **Model:** RandomForest Classifier (100 trees)
    
    **Features:**
    - Word count, sentence count, readability score
    - Top 50 TF-IDF dimensions
    
    **Performance:**
    - Accuracy: Shown in notebook output
    - F1-Score: Weighted average across classes
    
    **Limitations:**
    - TF-IDF may miss semantic paraphrases
    - Synthetic labels may not reflect true quality
    - Scraping requires stable internet connection
    
    ---
    
    ## API Usage (Python)
    
    ```python
    from streamlit_app.utils.analyzer import ContentAnalyzer
    
    # Initialize
    analyzer = ContentAnalyzer('models/', 'data/')
    
    # Analyze URL
    result = analyzer.analyze_url('https://example.com/article')
    
    print(result['quality_label'])  # High/Medium/Low
    print(result['similar_pages'])   # List of similar content
    ```
    
    ---
    
    ## Contact & Support
    
    For questions or issues, please refer to the GitHub repository.
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>SEO Content Analyzer v1.0 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
