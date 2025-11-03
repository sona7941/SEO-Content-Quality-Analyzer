"""
Quick script to run the pipeline and generate models
"""
import warnings
warnings.filterwarnings('ignore')

import os
import re
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests

import nltk
import textstat
from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import joblib

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Setup paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("REGENERATING MODELS WITH CURRENT SCIKIT-LEARN VERSION")
print("="*80)

# Load dataset
print("\n1. Loading data...")
df_raw = pd.read_csv(DATA_DIR / 'data.csv')
print(f"   Loaded {len(df_raw)} rows")

# Functions
def extract_title_and_body(html: str) -> Tuple[str, str]:
    if not html or pd.isna(html):
        return '', ''
    try:
        soup = BeautifulSoup(html, 'html.parser')
        title = ''
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        body_text = ''
        for tag_name in ['article', 'main']:
            element = soup.find(tag_name)
            if element:
                paragraphs = element.find_all('p')
                if paragraphs:
                    body_text = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
                    break
        if not body_text:
            paragraphs = soup.find_all('p')
            body_text = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
        body_text = re.sub(r'\s+', ' ', body_text).strip()
        return title, body_text
    except:
        return '', ''

def clean_text(text: str) -> str:
    if not text or pd.isna(text):
        return ''
    return re.sub(r'\s+', ' ', text).strip()

def safe_sentence_count(text: str) -> int:
    if not text or pd.isna(text):
        return 0
    try:
        return len(sent_tokenize(text))
    except:
        return 0

def safe_readability_score(text: str) -> float:
    if not text or pd.isna(text) or len(text.split()) == 0:
        return 0.0
    try:
        return textstat.flesch_reading_ease(text)
    except:
        return 0.0

# Parse HTML
print("\n2. Parsing HTML...")
parsed_data = []
for idx, row in df_raw.iterrows():
    title, body = extract_title_and_body(row.get('html_content', ''))
    body = clean_text(body)
    parsed_data.append({
        'url': row.get('url', ''),
        'title': title,
        'body_text': body,
        'word_count': len(body.split()) if body else 0
    })
df_extracted = pd.DataFrame(parsed_data)
df_extracted.to_csv(DATA_DIR / 'extracted_content.csv', index=False)
print(f"   Extracted {len(df_extracted)} pages")

# Feature engineering
print("\n3. Computing features...")
df_features = df_extracted.copy()
df_features['sentence_count'] = df_features['body_text'].apply(safe_sentence_count)
df_features['flesch_reading_ease'] = df_features['body_text'].apply(safe_readability_score)
df_features['is_thin'] = df_features['word_count'] < 500

# TF-IDF
corpus = df_features['body_text'].fillna('').tolist()
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', max_df=0.95, min_df=2)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
print(f"   TF-IDF matrix: {tfidf_matrix.shape}")

# Keywords
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
def extract_top_keywords(doc_vector, n=5):
    if doc_vector.nnz == 0:
        return ''
    values = doc_vector.toarray().ravel()
    top_indices = values.argsort()[::-1][:n]
    return '|'.join(feature_names[top_indices])

top_keywords_list = [extract_top_keywords(tfidf_matrix[i]) for i in range(tfidf_matrix.shape[0])]
df_features['top_keywords'] = top_keywords_list

df_features_output = df_features[['url', 'title', 'body_text', 'word_count', 
                                   'sentence_count', 'flesch_reading_ease', 'is_thin', 'top_keywords']].copy()
df_features_output.to_csv(DATA_DIR / 'features.csv', index=False)

# Save TF-IDF
joblib.dump(tfidf_vectorizer, MODELS_DIR / 'tfidf_vectorizer.pkl')
joblib.dump(tfidf_matrix, MODELS_DIR / 'tfidf_matrix.pkl')
print(f"   Saved TF-IDF models")

# Duplicate detection
print("\n4. Finding duplicates...")
similarity_matrix = cosine_similarity(tfidf_matrix)
duplicate_pairs = []
n_docs = similarity_matrix.shape[0]
for i in range(n_docs):
    for j in range(i + 1, n_docs):
        if similarity_matrix[i, j] >= 0.80:
            duplicate_pairs.append({
                'url1': df_features.loc[i, 'url'],
                'url2': df_features.loc[j, 'url'],
                'similarity': round(float(similarity_matrix[i, j]), 4)
            })
df_duplicates = pd.DataFrame(duplicate_pairs)
df_duplicates.to_csv(DATA_DIR / 'duplicates.csv', index=False)
print(f"   Found {len(df_duplicates)} duplicate pairs")

# Train classifier
print("\n5. Training classifier...")
def assign_quality_label(row):
    word_count = row['word_count']
    readability = row['flesch_reading_ease']
    if word_count > 1500 and 50 <= readability <= 70:
        return 'High'
    elif word_count < 500 or readability < 30:
        return 'Low'
    else:
        return 'Medium'

df_model = df_features.copy()
df_model['quality_label'] = df_model.apply(assign_quality_label, axis=1)

# Prepare features
n_tfidf = min(50, tfidf_matrix.shape[1])
X_tfidf = tfidf_matrix[:, :n_tfidf].toarray()
X_basic = df_model[['word_count', 'sentence_count', 'flesch_reading_ease']].fillna(0).values
X = np.hstack([X_basic, X_tfidf])
y = df_model['quality_label'].values

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n   Accuracy: {accuracy:.4f}")
print(f"   F1-Score: {f1:.4f}")

# Save model
joblib.dump(model, MODELS_DIR / 'quality_model.pkl')
print(f"\n   Model saved!")

print("\n" + "="*80)
print("DONE! All models regenerated with scikit-learn", __import__('sklearn').__version__)
print("="*80)
print("\nYou can now run: streamlit run streamlit_app/app.py")
