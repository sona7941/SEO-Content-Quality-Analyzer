"""
Modern SEO Analyzer with unique styling
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
    page_title="SEO Analyzer Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern gradient design
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Content container */
    .block-container {
        background: rgba(255, 255, 255, 0.97);
        border-radius: 20px;
        padding: 2.5rem !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
    }
    
    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        animation: fadeInDown 0.8s ease;
    }
    
    .sub-header {
        font-size: 1.4rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 300;
        animation: fadeInUp 0.8s ease;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Metric cards with gradient borders */
    .metric-card {
        background: linear-gradient(white, white) padding-box,
                    linear-gradient(135deg, #667eea 0%, #764ba2 100%) border-box;
        border: 3px solid transparent;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Quality labels with modern badges */
    .quality-high { 
        color: #fff;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
        animation: pulse 2s infinite;
    }
    .quality-medium { 
        color: #fff;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    .quality-low { 
        color: #fff;
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(250, 112, 154, 0.3);
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
        }
        50% {
            box-shadow: 0 4px 25px rgba(17, 153, 142, 0.5);
        }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-weight: 700;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.8rem 2.5rem;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        padding: 14px 28px;
        font-weight: 700;
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: #667eea;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 14px;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        font-weight: 700;
        padding: 1rem;
        border: 2px solid #e0e0e0;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Success/Info messages */
    .stSuccess, .stInfo {
        border-radius: 12px;
        border-left: 5px solid #38ef7d;
    }
    
    .stError, .stWarning {
        border-radius: 12px;
        border-left: 5px solid #f5576c;
    }
</style>
""", unsafe_allow_html=True)

# load analyzer
@st.cache_resource
def load_analyzer():
    return ContentAnalyzer(ROOT / 'models', ROOT / 'data')

analyzer = load_analyzer()

# Hero section with icon
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div style="font-size: 5rem; margin-bottom: 1rem; animation: fadeInDown 0.8s ease;">🎯</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">SEO Analyzer Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🚀 AI-Powered Content Analysis • Quality Scoring • Duplicate Detection</div>', unsafe_allow_html=True)

# Sidebar with modern design
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <h2 style="margin: 0; font-size: 2rem;">📊 Dashboard</h2>
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
        <h3 style="font-size: 1.5rem;">ℹ️ About</h3>
        <p style="font-size: 0.95rem; line-height: 1.8; margin-top: 1rem;">
        AI-powered content analysis with quality scoring, duplicate detection, and comprehensive SEO insights.
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
            "URL",
            placeholder="https://example.com/page",
            label_visibility="collapsed"
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
                st.markdown(f'<div class="metric-card"><strong>Quality Score</strong><br><span style="font-size: 2rem; font-weight: 800;">{result["quality_score"]:.1f}%</span></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><strong>Classification</strong><br><span class="{quality_class}">{result["quality_label"]}</span></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><strong>Word Count</strong><br><span style="font-size: 1.5rem;">{result["word_count"]}</span></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><strong>Readability</strong><br><span style="font-size: 1.5rem;">{result["readability"]:.1f}</span></div>', unsafe_allow_html=True)
            
            st.divider()
            
            # Content metrics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📝 Content Metrics")
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
                st.markdown("#### 🔗 HTML Elements")
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
                st.markdown("#### 🔎 Similar Content Found")
                for item in result['similar_content']:
                    with st.expander(f"Similarity: {item['similarity']:.1%} - {item['url']}"):
                        st.write(f"**Word Count:** {item['word_count']}")
                        st.write(f"**Quality:** {item['quality_label']}")
            else:
                st.info("ℹ️ No similar content found in the dataset")

# Tab 2: Dataset Insights
with tab2:
    st.markdown("### 📊 Content Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_quality_distribution(analyzer.df), use_container_width=True)
    with col2:
        fig = plot_duplicate_heatmap(analyzer.df_duplicates)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No duplicate pairs found")
    
    st.divider()
    
    # Top quality content
    st.markdown("### 🏆 Top Quality Content")
    top_content = analyzer.df.nlargest(5, 'quality_score')[['url', 'quality_score', 'quality_label', 'word_count']]
    
    for idx, row in top_content.iterrows():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            title = str(row.get('title', '')) if pd.notna(row.get('title')) else 'No title'
            title_short = title[:50] if len(title) > 50 else title
            st.write(f"**{title_short}**")
            st.caption(row['url'])
        with col2:
            quality_class = f"quality-{row['quality_label'].lower()}"
            st.markdown(f'<span class="{quality_class}">{row["quality_label"]}</span>', unsafe_allow_html=True)
        with col3:
            st.metric("Score", f"{row['quality_score']:.1f}%")
    
    st.divider()
    
    # Thin content
    st.markdown("### ⚠️ Thin Content Pages")
    thin_content = analyzer.df[analyzer.df['word_count'] < 300].nlargest(5, 'word_count')[['url', 'word_count', 'quality_score']]
    
    if not thin_content.empty:
        for idx, row in thin_content.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                title = str(row.get('title', '')) if pd.notna(row.get('title')) else 'No title'
                title_short = title[:50] if len(title) > 50 else title
                st.write(f"**{title_short}**")
                st.caption(row['url'])
            with col2:
                st.metric("Words", row['word_count'])
    else:
        st.success("✅ No thin content found!")

# Tab 3: Documentation
with tab3:
    st.markdown("### 📚 How It Works")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; border-radius: 15px; margin: 1rem 0;">
    <h4>🎯 Features</h4>
    <ul style="font-size: 1.05rem; line-height: 2;">
        <li><strong>Quality Scoring:</strong> ML-powered analysis of content quality</li>
        <li><strong>Duplicate Detection:</strong> Find similar/duplicate pages</li>
        <li><strong>SEO Metrics:</strong> Word count, readability, content structure</li>
        <li><strong>Content Analysis:</strong> Headings, links, images, keywords</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🔍 Quality Classification")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="quality-high">HIGH QUALITY</div>', unsafe_allow_html=True)
        st.write("• Comprehensive content")
        st.write("• Good structure")
        st.write("• Strong SEO signals")
    with col2:
        st.markdown('<div class="quality-medium">MEDIUM QUALITY</div>', unsafe_allow_html=True)
        st.write("• Adequate content")
        st.write("• Basic structure")
        st.write("• Room for improvement")
    with col3:
        st.markdown('<div class="quality-low">LOW QUALITY</div>', unsafe_allow_html=True)
        st.write("• Thin content")
        st.write("• Poor structure")
        st.write("• Needs work")
    
    st.divider()
    
    st.markdown("#### 📊 Metrics Explained")
    
    with st.expander("📈 Quality Score (0-100)"):
        st.write("""
        Machine learning model trained on content features:
        - Content length and depth
        - Readability and structure  
        - HTML elements and formatting
        - Keyword usage and relevance
        """)
    
    with st.expander("📝 Readability Score"):
        st.write("""
        Flesch Reading Ease score (0-100):
        - 90-100: Very easy to read
        - 60-70: Plain English
        - 0-30: Very difficult to read
        """)
    
    with st.expander("🔍 Duplicate Detection"):
        st.write("""
        Cosine similarity using TF-IDF vectors:
        - Compares content text similarity
        - Adjustable threshold (0.0 - 1.0)
        - Higher = more similar content
        """)
    
    st.divider()
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
    <h3>🚀 Ready to analyze?</h3>
    <p style="font-size: 1.1rem; margin-top: 1rem;">Head to the Analyze URL tab to get started!</p>
    </div>
    """, unsafe_allow_html=True)
