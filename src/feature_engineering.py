"""
feature_engineering.py — Feature Extraction & Engineering
==========================================================
TF-IDF, n-gram extraction, POS features, and composite feature matrix.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def build_tfidf_matrix(texts, max_features=5000, ngram_range=(1,1), min_df=5, max_df=0.95, sublinear_tf=True):
    """Build a TF-IDF matrix from text series. Returns (tfidf_matrix, fitted_vectorizer)."""
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                          min_df=min_df, max_df=max_df, sublinear_tf=sublinear_tf,
                          strip_accents='unicode', token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')
    matrix = vec.fit_transform(texts.fillna(''))
    return matrix, vec


def get_top_tfidf_terms(tfidf_matrix, vectorizer, group_labels, top_n=15):
    """Get top TF-IDF terms for each group (category, star rating, etc.)."""
    feature_names = vectorizer.get_feature_names_out()
    results = []
    for label in sorted(group_labels.unique()):
        mask = group_labels == label
        group_mean = tfidf_matrix[mask].mean(axis=0).A1
        top_indices = group_mean.argsort()[-top_n:][::-1]
        for idx in top_indices:
            results.append({'group': label, 'term': feature_names[idx], 'tfidf_score': group_mean[idx]})
    return pd.DataFrame(results)


def extract_ngrams(texts, n=2, top_k=20, min_freq=5):
    """Extract top n-grams from text series."""
    vec = CountVectorizer(ngram_range=(n, n), token_pattern=r'(?u)\b[a-zA-Z]{2,}\b', min_df=min_freq)
    matrix = vec.fit_transform(texts.fillna(''))
    freqs = matrix.sum(axis=0).A1
    names = vec.get_feature_names_out()
    return pd.DataFrame({'ngram': names, 'frequency': freqs}).sort_values('frequency', ascending=False).head(top_k).reset_index(drop=True)


def extract_ngrams_by_group(texts, group_labels, n=2, top_k=10):
    """Extract top n-grams for each group."""
    results = []
    for label in sorted(group_labels.unique()):
        mask = group_labels == label
        try:
            ngrams = extract_ngrams(texts[mask], n=n, top_k=top_k, min_freq=2)
            ngrams['group'] = label
            results.append(ngrams)
        except ValueError:
            continue
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def build_feature_matrix(df, text_column='cleaned_review', tfidf_features=500, include_meta=True):
    """Build comprehensive feature matrix combining TF-IDF and metadata features."""
    tfidf_matrix, tfidf_vec = build_tfidf_matrix(df[text_column], max_features=tfidf_features)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            columns=[f'tfidf_{n}' for n in tfidf_vec.get_feature_names_out()])
    if include_meta:
        meta = pd.DataFrame({
            'word_count': df.get('word_count', df[text_column].str.split().str.len()),
            'char_count': df.get('char_count', df[text_column].str.len()),
            'avg_word_length': df[text_column].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if isinstance(x, str) and x.strip() else 0),
            'exclamation_count': df[text_column].str.count('!'),
            'question_count': df[text_column].str.count(r'\?'),
            'uppercase_ratio': df[text_column].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)),
        })
        result = pd.concat([meta.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    else:
        result = tfidf_df
    return result, list(result.columns)
