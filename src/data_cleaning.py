"""
data_cleaning.py — Text Cleaning & Preprocessing Pipeline
==========================================================
Provides reusable functions for cleaning raw Trustpilot review text.
Used primarily by Notebook 01, but available to all notebooks.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import Optional


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'<[^>]+>', '', text)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\S+@\S+\.\S+', '', text)


def remove_special_characters(text: str, keep_punctuation: bool = False) -> str:
    """Remove special characters, optionally keeping basic punctuation."""
    if not isinstance(text, str):
        return ""
    if keep_punctuation:
        # Keep letters, numbers, spaces, and basic punctuation
        return re.sub(r'[^a-zA-Z0-9\s.,!?;:\'-]', '', text)
    else:
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()


def remove_repeated_chars(text: str, max_repeats: int = 3) -> str:
    """Reduce repeated characters (e.g., 'sooooo' -> 'sooo')."""
    if not isinstance(text, str):
        return ""
    pattern = r'(.)\1{' + str(max_repeats) + r',}'
    return re.sub(pattern, r'\1' * max_repeats, text)


def clean_text(text: str, lowercase: bool = True, keep_punctuation: bool = False) -> str:
    """
    Full cleaning pipeline for a single text string.
    
    Steps:
    1. Remove HTML tags
    2. Remove URLs
    3. Remove emails
    4. Remove special characters
    5. Reduce repeated characters
    6. Normalize whitespace
    7. Optionally lowercase
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_special_characters(text, keep_punctuation=keep_punctuation)
    text = remove_repeated_chars(text)
    text = normalize_whitespace(text)
    
    if lowercase:
        text = text.lower()
    
    return text


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str = 'review',
    output_column: str = 'cleaned_review',
    lowercase: bool = True,
    keep_punctuation: bool = True,
    drop_empty: bool = True,
    drop_duplicates: bool = True,
    min_length: int = 10
) -> pd.DataFrame:
    """
    Clean an entire DataFrame of reviews.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a text column
    text_column : str
        Name of the column containing raw review text
    output_column : str
        Name of the new column for cleaned text
    lowercase : bool
        Whether to lowercase the text
    keep_punctuation : bool
        Whether to keep basic punctuation (.,!? etc.)
    drop_empty : bool
        Whether to drop rows with empty cleaned text
    drop_duplicates : bool
        Whether to drop duplicate reviews
    min_length : int
        Minimum character length for a valid review
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with new cleaned text column
    """
    df = df.copy()
    
    # Apply cleaning
    df[output_column] = df[text_column].apply(
        lambda x: clean_text(x, lowercase=lowercase, keep_punctuation=keep_punctuation)
    )
    
    # Add word count column
    df['word_count'] = df[output_column].apply(lambda x: len(x.split()) if x else 0)
    
    # Add character count column
    df['char_count'] = df[output_column].apply(len)
    
    # Drop empty reviews
    if drop_empty:
        df = df[df['char_count'] >= min_length].reset_index(drop=True)
    
    # Drop duplicates based on cleaned text
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=[output_column]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} duplicate reviews")
    
    return df


def get_stopwords() -> set:
    """Get a comprehensive set of English stop words."""
    import nltk
    try:
        from nltk.corpus import stopwords
        nltk_stops = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        nltk_stops = set(stopwords.words('english'))
    
    # Add domain-specific stop words for review text
    custom_stops = {
        'would', 'could', 'also', 'get', 'got', 'one', 'two', 'even',
        'really', 'much', 'well', 'good', 'great', 'like', 'just',
        'know', 'us', 'go', 'going', 'went', 'come', 'came',
        'use', 'used', 'using', 'make', 'made', 'said', 'say',
        'back', 'still', 'thing', 'things', 'way', 'want', 'wanted',
        'company', 'review', 'reviews', 'trustpilot',
    }
    
    return nltk_stops | custom_stops


def tokenize_and_lemmatize(text: str, nlp=None) -> list:
    """
    Tokenize and lemmatize text using spaCy.
    
    Parameters
    ----------
    text : str
        Input text string
    nlp : spacy.Language, optional
        Pre-loaded spaCy model. If None, loads en_core_web_sm.
        
    Returns
    -------
    list
        List of lemmatized tokens (no stop words, no punctuation)
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    if nlp is None:
        import spacy
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 2
        and token.text.isalpha()
    ]
    
    return tokens
