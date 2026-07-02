"""
sentiment.py — Sentiment Analysis Helpers
===========================================
VADER, AFINN, and Transformer-based sentiment scoring.
"""

import pandas as pd
import numpy as np
from typing import Optional


def vader_sentiment(texts: pd.Series) -> pd.DataFrame:
    """Apply VADER sentiment analysis. Returns DataFrame with neg, neu, pos, compound."""
    import nltk
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    sia = SentimentIntensityAnalyzer()
    scores = texts.fillna('').apply(lambda x: sia.polarity_scores(x))
    return pd.DataFrame(scores.tolist())


def afinn_sentiment(texts: pd.Series) -> pd.Series:
    """Apply AFINN sentiment scoring. Returns a Series of scores."""
    from afinn import Afinn
    afinn = Afinn()
    return texts.fillna('').apply(afinn.score)


def transformer_sentiment(texts: pd.Series, batch_size: int = 32, max_length: int = 512) -> pd.DataFrame:
    """Apply transformer-based sentiment (distilbert). Returns DataFrame with label and score."""
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",
                          truncation=True, max_length=max_length)
    
    results = []
    text_list = texts.fillna('').tolist()
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        batch = [t[:max_length*4] if t else "empty" for t in batch]  # rough char limit
        preds = classifier(batch)
        results.extend(preds)
    
    df = pd.DataFrame(results)
    df.columns = ['transformer_label', 'transformer_score']
    # Convert to numeric: POSITIVE = +score, NEGATIVE = -score
    df['transformer_compound'] = df.apply(
        lambda r: r['transformer_score'] if r['transformer_label'] == 'POSITIVE' else -r['transformer_score'], axis=1)
    return df


def classify_sentiment(score: float, pos_threshold: float = 0.05, neg_threshold: float = -0.05) -> str:
    """Classify a compound score into Positive/Neutral/Negative."""
    if score >= pos_threshold:
        return 'Positive'
    elif score <= neg_threshold:
        return 'Negative'
    return 'Neutral'
