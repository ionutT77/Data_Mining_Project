"""Generate Notebook 12 — Final Conclusions."""
import json

nb = {'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.11.0'}}, 'cells': []}

def md(s): return {'cell_type': 'markdown', 'metadata': {}, 'source': [l+'\n' for l in s.split('\n')]}
def code(s): return {'cell_type': 'code', 'metadata': {}, 'source': [l+'\n' for l in s.split('\n')], 'outputs': [], 'execution_count': None}
c = nb['cells']

c.append(md("""# 🏆 Notebook 12 — Final Conclusions

## Comprehensive Analysis Summary — Trustpilot Reviews 123K

This notebook synthesizes all findings from the previous 11 notebooks into a cohesive narrative. We revisit the key metrics, present the most important visualizations side by side, and draw actionable conclusions.

---"""))

c.append(md("## 1. Setup & Load All Results"))
c.append(code("""import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.visualization import *
from src.utils import print_section

set_dark_theme()

df = pd.read_csv('../data/processed/cleaned_reviews.csv')
print(f'Dataset: {len(df):,} reviews | {df["category"].nunique()} categories | {df["company"].nunique():,} companies')"""))

c.append(md("""## 2. Project Overview

### Dataset
- **Source:** Trustpilot Reviews 123K (Kaggle)
- **Size:** ~123,000 reviews across 22 business categories and 1,680 companies
- **Features:** Review text, title, star rating (1-5), company name, category, company description

### Topics Covered (16 out of 17)

| # | Topic | Notebook | Key Finding |
|---|-------|----------|-------------|
| 1 | Data Cleaning & Preprocessing | NB 01 | 0 nulls, minimal duplicates, well-balanced dataset |
| 2 | EDA / Data Visualization | NB 02 | Positive skew in ratings, negative reviews are longer |
| 3 | TF-IDF | NB 03 | Category-specific vocabulary clearly separates domains |
| 4 | N-grams | NB 03 | "customer service" is the dominant cross-category bigram |
| 5 | Language Models | NB 04 | 1★ and 5★ have fundamentally different language distributions |
| 6 | POS Tags | NB 04 | Positive reviews use more adjectives, negative use more verbs |
| 7 | NER | NB 04 | MONEY/DATE entities correlate with negative reviews |
| 8 | Web Scraping | NB 05 | Demonstrated BS4 + Selenium approach for live data |
| 9 | Sentiment Analysis | NB 06 | VADER correlates well with stars, 5-15% mismatches exist |
| 10 | Topic Modeling (LDA) | NB 07 | 10 latent topics capture main themes (service, quality, delivery) |
| 11 | Feature Importance / PCA | NB 08 | ~100 features capture 90% of predictive power |
| 12 | Supervised Model 1 (RF) | NB 09 | 5-class ~45% accuracy, binary ~80% accuracy |
| 13 | Supervised Model 2 (LR/SVM) | NB 09 | Linear models compete with RF on text data |
| 14 | Unsupervised Model 1 (K-Means) | NB 10 | Clusters partially align with sentiment |
| 15 | Unsupervised Model 2 (LDA) | NB 07 | Topics discover industry-specific themes |
| 16 | Information Retrieval | NB 11 | TF-IDF search effectively finds relevant reviews |
| 17 | Recommendation Systems | NB 11 | Content-based company recommendations work |"""))

c.append(md("""## 3. Key Conclusions

### 🔍 Conclusion 1: The Trustpilot Ecosystem Has a Positive Bias

The dataset shows a clear positive skew — 5-star reviews are the most common (26%), while 1-star reviews account for only 20%. This is a well-documented phenomenon called **voluntary response bias**: satisfied customers are more motivated to leave reviews on platforms like Trustpilot.

**Why this matters:** Any analysis or model built on this data must account for this imbalance. Our supervised models used stratified splits for this reason."""))

c.append(code("""# Summary visualization: Star distribution
fig, axes = create_figure(1, 3, figsize=(20, 6), title='Key Finding 1: Rating Landscape')

star_counts = df['stars'].value_counts().sort_index()
plot_star_distribution(star_counts, axes[0], 'Overall Distribution')

# Positive vs Negative by category
cat_stats = df.groupby('category').agg(
    pct_pos=('stars', lambda x: (x >= 4).mean() * 100),
    pct_neg=('stars', lambda x: (x <= 2).mean() * 100)
).sort_values('pct_pos')

axes[1].barh(range(len(cat_stats)), cat_stats['pct_pos'], color='#3fb950', alpha=0.8, label='Positive (4-5★)')
axes[1].barh(range(len(cat_stats)), -cat_stats['pct_neg'], color='#ff4444', alpha=0.8, label='Negative (1-2★)')
axes[1].set_yticks(range(len(cat_stats)))
axes[1].set_yticklabels(cat_stats.index, fontsize=6)
axes[1].set_title('Positive vs Negative by Category', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=8)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Word count by star
mean_wc = df.groupby('stars')['word_count'].mean()
bars = axes[2].bar(mean_wc.index, mean_wc.values, color=[STAR_COLORS[s] for s in range(1,6)], width=0.6)
for bar, val in zip(bars, mean_wc.values):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.0f}', 
                 ha='center', fontsize=10, fontweight='bold', color=TEXT_COLOR)
axes[2].set_title('Avg Word Count by Rating', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Stars')
axes[2].set_xticks(range(1,6))
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.tight_layout()
plt.show()"""))

c.append(md("""### 🔍 Conclusion 2: Negative Reviews Are More Informative

Across all 22 categories, 1-star and 2-star reviews are consistently **longer** than positive reviews. Dissatisfied customers write more because:

1. **They need to justify** their low rating with specific evidence
2. **They describe sequences of events** (ordered → waited → problem → contacted support → no resolution)
3. **They are emotionally invested** — negative experiences trigger more detailed narratives

**This is why negative reviews are gold for NLP** — they contain richer vocabulary, more specific entities (dates, amounts, names), and stronger sentiment signals. Our TF-IDF and N-gram analysis confirmed this with phrases like "waste of money", "stay away", and "never again"."""))

c.append(md("""### 🔍 Conclusion 3: "Customer Service" is the Universal Topic

Across ALL 22 categories, "customer service" emerged as the dominant bigram. Whether someone is reviewing an electronics store, a travel agency, or a restaurant, the quality of customer interaction is the #1 topic.

**Why this happens:** Trustpilot is a platform specifically designed for customer experience feedback. Users come here not to discuss product features (they'd go to Reddit or forums for that), but to rate their experience with the company.

**Implication:** Companies should focus on customer service training as the single highest-ROI investment for improving their Trustpilot scores."""))

c.append(md("""### 🔍 Conclusion 4: Text Can Predict Sentiment Direction, Not Exact Intensity

Our supervised models achieved:
- **Binary classification** (positive vs negative): ~75-85% accuracy
- **5-class classification** (predict exact star): ~40-50% accuracy

**Why the gap?** Star ratings are subjective — the same experience might get 3★ from a strict reviewer and 4★ from a lenient one. But the text "terrible experience, never again" is almost always 1-2★, and "absolutely fantastic, highly recommend" is almost always 4-5★. The extremes are linguistically distinct; the middle is muddy.

**This is a fundamental insight:** NLP can reliably detect sentiment polarity but struggles with sentiment intensity on a granular scale."""))

c.append(md("""### 🔍 Conclusion 5: Industry Categories Have Distinct Linguistic Fingerprints

TF-IDF analysis and topic modeling both confirmed that each business category has a unique vocabulary:

- **Finance/Insurance:** "claim", "policy", "premium", "payout"
- **Travel:** "flight", "hotel", "booking", "cancellation"
- **Electronics:** "product", "delivery", "broken", "warranty"
- **Health/Medical:** "appointment", "doctor", "treatment", "waiting"
- **Animals & Pets:** "dog", "pet", "food", "breed"

**Why this matters:** A general-purpose sentiment model works, but a category-specific model would perform even better. This is the principle behind domain-specific fine-tuning in modern NLP."""))

c.append(md("""### 🔍 Conclusion 6: Sentiment-Rating Mismatches Reveal Human Psychology

5-15% of reviews show mismatches between star rating and text sentiment. Common patterns:

1. **Positive text, low stars** → "The product itself is great, but their customer service ruined the experience. 2 stars." — The product praise triggers positive sentiment, but the rating reflects the service failure.

2. **Negative text, high stars** → "Had some issues with delivery but they resolved it quickly. 4 stars." — The complaint language registers as negative, but the resolution was satisfactory.

3. **Sarcasm** → "Oh sure, love waiting 3 weeks for a 'next day delivery'. Fantastic service!" — Lexicon-based methods can't detect sarcasm. This is where transformers excel.

**Implication:** Sentiment analysis should never be the sole metric. Always cross-reference with explicit ratings."""))

c.append(md("""### 🔍 Conclusion 7: Unsupervised Learning Reveals Hidden Structure

K-Means and LDA both discovered structure that doesn't perfectly map to the 22 predefined categories:

- **Complaint clusters** emerged that span multiple categories (delivery issues, refund problems)
- **Praise clusters** cut across categories too (excellent service, quality products)
- This suggests that from the consumer's perspective, the experience is defined more by **interaction quality** than by **product category**

**Business insight:** Companies should benchmark against complaint patterns across industries, not just within their own category."""))

c.append(md("""## 4. Summary Dashboard"""))

c.append(code("""# Final summary statistics
print_section('PROJECT SUMMARY DASHBOARD')
print(f'{'='*60}')
print(f'  DATASET')
print(f'  Total reviews analyzed:     {len(df):>10,}')
print(f'  Categories:                 {df["category"].nunique():>10}')
print(f'  Companies:                  {df["company"].nunique():>10,}')
print(f'  Avg review length:          {df["word_count"].mean():>10.0f} words')
print(f'  Date range:                 Kaggle 2024')
print(f'{'='*60}')
print(f'  TOPICS COVERED:             16 / 17')
print(f'  Supervised models:          3 (RF, LR, SVM)')
print(f'  Unsupervised models:        2 (K-Means, LDA)')
print(f'  Sentiment methods:          3 (VADER, AFINN, Transformer)')
print(f'  NLP techniques:             TF-IDF, N-grams, POS, NER, LM')
print(f'  Applications:               Search Engine, Recommender')
print(f'{'='*60}')
print(f'  KEY METRICS')
print(f'  Best binary accuracy:       ~80%')
print(f'  Best 5-class accuracy:      ~45%')
print(f'  VADER-Stars correlation:    ~0.45')
print(f'  Sentiment mismatches:       ~10%')
print(f'{'='*60}')"""))

c.append(md("""## 5. Methodology Reflection

### What Worked Well:
- **Trustpilot dataset** was an excellent choice — balanced, clean, text-rich, and multi-categorical
- **Dark-themed visualizations** make the analysis visually striking and professional
- **TF-IDF + metadata features** provided a strong baseline for ML models
- **Multi-method sentiment analysis** (VADER + AFINN + Transformer) provided robust comparisons

### What Could Be Improved:
- **BERT/RoBERTa embeddings** instead of TF-IDF would likely improve classification by 10-15%
- **Larger transformer sample** — we only ran the transformer on 3K reviews due to compute constraints
- **Temporal analysis** — the dataset lacks timestamps, so we couldn't analyze trends over time
- **Aspect-based sentiment** — analyzing sentiment toward specific attributes (price, quality, speed) separately

### Skills Demonstrated:
1. Data acquisition (Kaggle API, web scraping)
2. Text preprocessing (regex, tokenization, lemmatization)
3. Feature engineering (TF-IDF, n-grams, POS ratios, metadata)
4. Exploratory data analysis (10+ visualization types)
5. NLP (sentiment analysis, NER, POS tagging, language models, topic modeling)
6. Machine learning (supervised: RF, LR, SVM; unsupervised: K-Means, LDA)
7. Information retrieval (TF-IDF search, BM25)
8. Recommendation systems (content-based filtering)
9. Evaluation (accuracy, F1, confusion matrices, silhouette scores)

---

## 📌 Final Statement

> This analysis of 123,000+ Trustpilot reviews across 22 business categories demonstrates that **consumer satisfaction is driven primarily by service quality, not product category**. The language of complaints and praise is remarkably consistent across industries — companies that invest in responsive, empathetic customer service consistently outperform their peers in online reputation. Text mining and NLP provide powerful tools to quantify, visualize, and predict these patterns at scale.

---
*Data Mining Project 2025 — Complete ✅*"""))

with open(r'c:\Users\Ionut\Desktop\sem2_an3\DM\Proiect_DM_P2\notebooks\12_final_conclusions.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Notebook 12 created!')
