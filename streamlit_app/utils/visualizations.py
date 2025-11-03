"""
Visualization utilities for Streamlit app
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any


def plot_quality_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing quality label distribution.
    
    Args:
        df: DataFrame with features including quality_label column
        
    Returns:
        Plotly figure
    """
    if df.empty:
        return go.Figure()
    
    # Count quality labels from the full features dataframe
    quality_counts = df['quality_label'].value_counts().reset_index()
    quality_counts.columns = ['quality_label', 'count']
    
    colors = {
        'High': '#28a745',
        'Medium': '#ffc107',
        'Low': '#dc3545'
    }
    
    fig = px.bar(
        quality_counts,
        x='quality_label',
        y='count',
        title='Content Quality Distribution',
        labels={'quality_label': 'Quality Level', 'count': 'Number of Pages'},
        color='quality_label',
        color_discrete_map=colors
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Quality Level",
        yaxis_title="Number of Pages"
    )
    
    return fig


def plot_duplicate_heatmap(similarity_matrix: Any) -> go.Figure:
    """
    Create a heatmap of content similarity.
    
    Args:
        similarity_matrix: Numpy array or sparse matrix
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        colorscale='Viridis',
        colorbar=dict(title='Similarity')
    ))
    
    fig.update_layout(
        title='Content Similarity Heatmap',
        height=500,
        xaxis_title='Document Index',
        yaxis_title='Document Index'
    )
    
    return fig


def create_metrics_display(metrics: Dict[str, Any]) -> str:
    """
    Create formatted HTML for metrics display.
    
    Args:
        metrics: Dictionary of metric name-value pairs
        
    Returns:
        HTML string
    """
    html = '<div style="display: flex; gap: 1rem;">'
    
    for name, value in metrics.items():
        html += f'''
        <div style="
            flex: 1;
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        ">
            <div style="font-size: 0.9rem; color: #666;">{name}</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{value}</div>
        </div>
        '''
    
    html += '</div>'
    return html
