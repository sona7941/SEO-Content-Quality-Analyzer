"""
Content analysis utilities
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import joblib
import requests
from bs4 import BeautifulSoup
import re
import textstat
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity


class ContentAnalyzer:
    """Handles content quality analysis and similarity detection"""
    
    def __init__(self, models_dir: Path, data_dir: Path):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # load models
        self.vectorizer = joblib.load(self.models_dir / 'tfidf_vectorizer.pkl')
        self.tfidf_matrix = joblib.load(self.models_dir / 'tfidf_matrix.pkl')
        self.classifier = joblib.load(self.models_dir / 'quality_model.pkl')
        
        # load data
        self.df_extracted = pd.read_csv(self.data_dir / 'extracted_content.csv')
        self.df_features = pd.read_csv(self.data_dir / 'features.csv')
        
        # Add quality predictions if not already present
        if 'quality_label' not in self.df_features.columns:
            # Prepare features for prediction (same as training)
            feature_cols = ['word_count', 'sentence_count', 'flesch_reading_ease']
            X_basic = self.df_features[feature_cols].fillna(0).values
            
            # Get TF-IDF features (first 50 columns to match training)
            n_tfidf = min(50, self.tfidf_matrix.shape[1])
            X_tfidf = self.tfidf_matrix[:, :n_tfidf].toarray()
            
            # Combine features
            X = np.hstack([X_basic, X_tfidf])
            
            # Get predictions
            predictions = self.classifier.predict(X)
            probabilities = self.classifier.predict_proba(X)
            
            # Add to dataframe
            self.df_features['quality_label'] = predictions
            self.df_features['quality_score'] = probabilities.max(axis=1) * 100
        
        # load duplicates if exists
        duplicates_path = self.data_dir / 'duplicates.csv'
        if duplicates_path.exists():
            try:
                self.df_duplicates = pd.read_csv(duplicates_path)
            except pd.errors.EmptyDataError:
                # File exists but is empty (no duplicates found)
                self.df_duplicates = pd.DataFrame(columns=['url1', 'url2', 'similarity'])
        else:
            self.df_duplicates = pd.DataFrame(columns=['url1', 'url2', 'similarity'])
    
    def scrape_url(self, url: str, timeout: int = 10) -> str:
        """Scrape HTML from URL"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return ''
    
    def extract_title_and_body(self, html: str) -> tuple:
        """Extract title and body from HTML"""
        if not html or pd.isna(html):
            return '', ''
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # get title
            title = ''
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            
            # get body - try semantic tags first
            body_text = ''
            for tag_name in ['article', 'main']:
                element = soup.find(tag_name)
                if element:
                    paragraphs = element.find_all('p')
                    if paragraphs:
                        body_text = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
                        break
            
            # fallback to all paragraphs
            if not body_text:
                paragraphs = soup.find_all('p')
                body_text = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
            
            body_text = re.sub(r'\s+', ' ', body_text).strip()
            return title, body_text
            
        except Exception:
            return '', ''
    
    def clean_text(self, text: str) -> str:
        """Clean text"""
        if not text or pd.isna(text):
            return ''
        return re.sub(r'\s+', ' ', text).strip()
    
    def safe_sentence_count(self, text: str) -> int:
        """Count sentences"""
        if not text or pd.isna(text):
            return 0
        try:
            return len(sent_tokenize(text))
        except:
            return 0
    
    def safe_readability_score(self, text: str) -> float:
        """Compute readability"""
        if not text or pd.isna(text) or len(text.split()) == 0:
            return 0.0
        try:
            return textstat.flesch_reading_ease(text)
        except:
            return 0.0
    
    def analyze_url(self, url: str, similarity_threshold: float = 0.4) -> Dict[str, Any]:
        """Analyze URL for quality and similarity"""
        # scrape
        html = self.scrape_url(url)
        if not html:
            return {
                'url': url,
                'error': 'Failed to scrape URL. Please check the URL and try again.'
            }
        
        # extract content
        title, body = self.extract_title_and_body(html)
        body = self.clean_text(body)
        
        if not body:
            return {
                'url': url,
                'error': 'No content found. Page may be empty or protected.'
            }
        
        # compute features
        word_count = len(body.split())
        sentence_count = self.safe_sentence_count(body)
        readability = self.safe_readability_score(body)
        is_thin = word_count < 500
        
        # vectorize
        tfidf_vector = self.vectorizer.transform([body])
        
        # find similar pages
        similarities = cosine_similarity(tfidf_vector, self.tfidf_matrix).ravel()
        top_indices = similarities.argsort()[::-1][:5]
        
        similar_pages = []
        for idx in top_indices:
            sim_score = similarities[idx]
            if sim_score > similarity_threshold:
                similar_pages.append({
                    'url': self.df_extracted.loc[idx, 'url'],
                    'title': self.df_extracted.loc[idx, 'title'],
                    'similarity': round(float(sim_score), 4)
                })
        
        # predict quality
        n_tfidf = min(50, tfidf_vector.shape[1])
        X_tfidf_features = tfidf_vector[:, :n_tfidf].toarray()
        X_basic_features = np.array([[word_count, sentence_count, readability]])
        X_combined = np.hstack([X_basic_features, X_tfidf_features])
        
        quality_label = self.classifier.predict(X_combined)[0]
        quality_proba = self.classifier.predict_proba(X_combined)[0]
        quality_score = float(quality_proba.max() * 100)
        
        # Get additional metrics for the result
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        paragraph_count = len(soup.find_all('p'))
        image_count = len(soup.find_all('img'))
        link_count = len(soup.find_all('a'))
        heading_count = len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
        
        words = body.split()
        unique_words = len(set(words))
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        
        # Format similar pages to match expected structure
        similar_content = []
        for page in similar_pages:
            page_idx = self.df_features[self.df_features['url'] == page['url']].index
            if len(page_idx) > 0:
                page_data = self.df_features.loc[page_idx[0]]
                similar_content.append({
                    'url': page['url'],
                    'similarity': page['similarity'],
                    'word_count': int(page_data['word_count']),
                    'quality_label': page_data.get('quality_label', 'Unknown')
                })
        
        return {
            'url': url,
            'title': title,
            'word_count': int(word_count),
            'sentence_count': int(sentence_count),
            'paragraph_count': int(paragraph_count),
            'readability': round(float(readability), 2),
            'quality_score': round(quality_score, 1),
            'quality_label': quality_label,
            'image_count': int(image_count),
            'link_count': int(link_count),
            'heading_count': int(heading_count),
            'unique_words': int(unique_words),
            'avg_words_per_sentence': round(avg_words_per_sentence, 1),
            'is_thin_content': bool(is_thin),
            'similar_content': similar_content
        }
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get basic stats"""
        thin_count = (self.df_features['word_count'] < 500).sum()
        
        return {
            'total_pages': len(self.df_features),
            'duplicate_pairs': len(self.df_duplicates),
            'thin_content_pct': (thin_count / len(self.df_features)) * 100
        }
    
    def get_quality_distribution(self) -> pd.DataFrame:
        """Get quality label distribution"""
        if 'quality_label' in self.df_features.columns:
            return self.df_features['quality_label'].value_counts().reset_index()
        return pd.DataFrame()
    
    def get_feature_statistics(self) -> pd.DataFrame:
        """Get feature stats"""
        cols = ['word_count', 'sentence_count', 'flesch_reading_ease']
        return self.df_features[cols].describe().round(2)
    
    def get_duplicates(self) -> pd.DataFrame:
        return self.df_duplicates
    
    def get_top_content(self, n: int = 5) -> List[Dict]:
        """Get highest quality content"""
        top = self.df_features.nlargest(n, 'word_count')
        return top[['url', 'title', 'word_count', 'flesch_reading_ease']].to_dict('records')
    
    def get_thin_content(self, n: int = 5) -> List[Dict]:
        """Get thin content"""
        thin = self.df_features[self.df_features['word_count'] < 500].head(n)
        return thin[['url', 'title', 'word_count', 'flesch_reading_ease']].to_dict('records')
    
    def get_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        word_count = result['word_count']
        readability = result['flesch_reading_ease']
        quality = result['quality_label']
        
        # word count
        if word_count < 500:
            recommendations.append("WARNING: Content is too thin. Aim for at least 500 words for better SEO.")
        elif word_count < 1000:
            recommendations.append("Consider expanding content to 1000+ words for improved rankings.")
        elif word_count > 3000:
            recommendations.append("Consider breaking long content into multiple focused pages.")
        else:
            recommendations.append("Word count is in the optimal range for SEO.")
        
        # readability
        if readability < 30:
            recommendations.append("Content is very difficult to read. Simplify sentences and vocabulary.")
        elif readability < 50:
            recommendations.append("Content readability could be improved for broader audience appeal.")
        elif readability > 80:
            recommendations.append("Content may be too simple. Consider adding more depth and detail.")
        else:
            recommendations.append("Readability is in the optimal range (50-80).")
        
        # quality
        if quality == 'Low':
            recommendations.append("Overall quality is low. Focus on depth, readability, and uniqueness.")
        elif quality == 'Medium':
            recommendations.append("Content quality is acceptable but has room for improvement.")
        else:
            recommendations.append("Excellent content quality! Maintain these standards.")
        
        # similarity
        if result.get('similar_pages'):
            recommendations.append("WARNING: Similar content detected. Ensure your content offers unique value.")
        
        return recommendations
