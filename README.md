#  Data Mining Project 2025 — Trustpilot Reviews Analysis

## Overview
A comprehensive data mining analysis of **123,000+ Trustpilot reviews** across **22 business categories** and **1,680 companies**. This project demonstrates 16 out of 17 topics from the Data Mining course specification.

## Dataset
- **Source:** [Trustpilot Reviews 123K (Kaggle)](https://www.kaggle.com/datasets/jerassy/trustpilot-reviews-123k)
- **Size:** 123,181 reviews
- **Features:** category, company, description, title, review text, star rating (1-5)

## Topics Covered (16/17)
| Topic | Notebook |
|-------|----------|
| Data Cleaning & Preprocessing | `01_data_loading_and_cleaning.ipynb` |
| EDA / Data Visualization | `02_eda_and_visualization.ipynb` |
| TF-IDF | `03_text_analysis_tfidf_ngrams.ipynb` |
| N-grams | `03_text_analysis_tfidf_ngrams.ipynb` |
| Language Models | `04_pos_ner_language_models.ipynb` |
| POS Tags | `04_pos_ner_language_models.ipynb` |
| NER | `04_pos_ner_language_models.ipynb` |
| Web Scraping | `05_web_scraping.ipynb` |
| Sentiment Analysis | `06_sentiment_analysis.ipynb` |
| Topic Modeling (LDA) | `07_topic_modeling.ipynb` |
| Feature Importance / PCA | `08_feature_importance_pca.ipynb` |
| Supervised Model 1 (Random Forest) | `09_supervised_learning.ipynb` |
| Supervised Model 2 (Logistic Regression / SVM) | `09_supervised_learning.ipynb` |
| Unsupervised Model 1 (K-Means) | `10_unsupervised_learning.ipynb` |
| Unsupervised Model 2 (LDA) | `07_topic_modeling.ipynb` |
| Information Retrieval | `11_information_retrieval_recommendations.ipynb` |
| Recommendation Systems | `11_information_retrieval_recommendations.ipynb` |

## Project Structure
```
Proiect_DM_P2/
├── data/
│   ├── raw/                    # Original Kaggle CSV
│   ├── processed/              # Cleaned, feature-engineered data
│   └── models/                 # Saved model artifacts
├── src/
│   ├── data_cleaning.py        # Text cleaning pipeline
│   ├── feature_engineering.py  # TF-IDF, n-grams, feature matrix
│   ├── sentiment.py            # VADER, AFINN, Transformer sentiment
│   ├── visualization.py        # Premium dark-themed plotting
│   └── utils.py                # Path helpers, utilities
├── notebooks/                  # 12 analysis notebooks (run in order)
├── requirements.txt
└── README.md
```

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Download spaCy model: `python -m spacy download en_core_web_sm`
3. Run notebooks in order (01 → 12) from the `notebooks/` directory

## Key Findings
1. **Customer service is king** — "customer service" is the #1 topic across all 22 categories
2. **Negative reviews are more informative** — 1★ reviews are 40% longer than 5★ reviews
3. **Text predicts sentiment direction, not intensity** — Binary (positive/negative) achieves ~80% accuracy; 5-class ~45%
4. **5-15% sentiment-rating mismatches** — Caused by sarcasm, mixed experiences, and cultural rating differences
5. **Industry fingerprints exist** — Each category has distinctive TF-IDF vocabulary

## Author
Data Mining 2025 — Solo Project
