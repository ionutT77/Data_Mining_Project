"""
utils.py — General Utilities
==============================
Path helpers, timing, and miscellaneous functions.
"""

import os
import time
import functools
import pandas as pd

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
DATA_MODELS = os.path.join(PROJECT_ROOT, 'data', 'models')
DATA_FIGURES = os.path.join(PROJECT_ROOT, 'data', 'figures')


def timer(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  ⏱ {func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def load_raw_data(filename='trustpilot_reviews.csv'):
    """Load raw dataset from data/raw/."""
    path = os.path.join(DATA_RAW, filename)
    return pd.read_csv(path)


def load_processed_data(filename='cleaned_reviews.csv'):
    """Load processed dataset from data/processed/."""
    path = os.path.join(DATA_PROCESSED, filename)
    return pd.read_csv(path)


def save_processed_data(df, filename='cleaned_reviews.csv'):
    """Save processed dataset to data/processed/."""
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    path = os.path.join(DATA_PROCESSED, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(df):,} rows)")


def ensure_dirs():
    """Create all required directories."""
    for d in [DATA_RAW, DATA_PROCESSED, DATA_MODELS, DATA_FIGURES]:
        os.makedirs(d, exist_ok=True)


def print_section(title: str, char='═', width=70):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n── {title} {'─' * (60 - len(title))}\n")
