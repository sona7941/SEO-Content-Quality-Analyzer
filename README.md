# SEO Content Quality Analyzer

ML pipeline for analyzing web content quality and finding duplicate pages.

## What it does
- Scores content quality (Low/Medium/High)
- Finds duplicate/similar content using TF-IDF
- Calculates readability metrics
- Gives SEO recommendations
- Web interface with Streamlit

## Setup

Clone and install dependencies:

```powershell
# Using conda (recommended for Windows)
conda create -n seo-env python=3.10 -y
conda activate seo-env
conda install numpy scipy pandas scikit-learn joblib -c conda-forge -y
pip install -r requirements.txt
```

Or use pip:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to run

### Jupyter notebook
Run the pipeline to train models and process data:

```powershell
jupyter notebook notebooks/seo_pipeline.ipynb
```

This generates all the model files and processed datasets.

### Streamlit app
After running the notebook, launch the web interface:

```powershell
streamlit run streamlit_app/app.py
```

Opens at `http://localhost:8501`

## Implementation notes

### HTML parsing
Using BeautifulSoup with lxml parser. Checks for `<article>` and `<main>` tags first, falls back to all `<p>` tags if needed. Works pretty well for most pages.

### Features
- Basic: word count, sentences, Flesch readability
- TF-IDF: top 50 dimensions (5000 vocab max)
- Also extract top 5 keywords per page

### Duplicate detection
Cosine similarity on TF-IDF vectors, threshold at 0.80 (pretty strict to avoid false positives)

### Quality model
RandomForest with 100 trees. Simple synthetic labels based on word count + readability:
- High: >1500 words, readability 50-70
- Low: <500 words OR readability <30  
- Medium: everything else

Not perfect but works reasonably well.

## Deploying online

Push to GitHub, then go to [share.streamlit.io](https://share.streamlit.io) and connect your repo. Set main file to `streamlit_app/app.py` and you're good to go.

## Known issues
- TF-IDF doesn't catch semantic paraphrasing (would need embeddings like sentence-transformers)
- Readability scores can be weird on short/noisy text
- Labels are synthetic, not based on real quality metrics
- Some sites block scraping

## Possible improvements
- Better duplicate detection with semantic embeddings
- More visualizations (word clouds, heatmaps, etc)
- Sentiment analysis, NER
- REST API with FastAPI
- Batch processing for large datasets

