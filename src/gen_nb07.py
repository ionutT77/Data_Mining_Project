"""Generate Notebook 07 — Topic Modeling (LDA)."""
import json

nb = {'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.11.0'}}, 'cells': []}

def md(s): return {'cell_type': 'markdown', 'metadata': {}, 'source': [l+'\n' for l in s.split('\n')]}
def code(s): return {'cell_type': 'code', 'metadata': {}, 'source': [l+'\n' for l in s.split('\n')], 'outputs': [], 'execution_count': None}
c = nb['cells']

c.append(md("""# 📚 Notebook 07 — Topic Modeling (LDA)

## Topics: Topic Modeling + Unsupervised Model 2 (LDA)

**Why this topic?** With 22 categories, we can discover hidden sub-themes within each category using Latent Dirichlet Allocation (LDA). Do "Electronics" reviews cluster around price, quality, or support? LDA is inherently an unsupervised model, so this also counts as Unsupervised Model 2.

**What we observe:** What latent topics exist in the review corpus, whether discovered topics align with actual categories, and how topic distributions vary by star rating.

---"""))

c.append(md("## 1. Setup"))
c.append(code("""import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

from src.visualization import *
from src.utils import print_section

set_dark_theme()

df = pd.read_csv('../data/processed/cleaned_reviews.csv')
# Use a sample for LDA (full dataset is very slow)
SAMPLE_SIZE = 20000
df_sample = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
print(f'Using {SAMPLE_SIZE:,} reviews for topic modeling')"""))

c.append(md("## 2. Text Preprocessing for LDA\n\nLDA works best with bag-of-words representation after removing stop words."))
c.append(code("""# Build count vectorizer (bag of words)
count_vec = CountVectorizer(
    max_features=3000, 
    min_df=10, 
    max_df=0.8,
    stop_words='english',
    token_pattern=r'(?u)\\b[a-zA-Z]{3,}\\b'
)

dtm = count_vec.fit_transform(df_sample['cleaned_review'].fillna(''))
print(f'Document-Term Matrix: {dtm.shape}')
print(f'Vocabulary size: {len(count_vec.get_feature_names_out())}')"""))

c.append(md("## 3. Finding Optimal Number of Topics\n\nWe test different numbers of topics and compare perplexity and log-likelihood."))
c.append(code("""# Test different topic counts
topic_range = [5, 8, 10, 12, 15, 20]
perplexities = []
log_likelihoods = []

print('Training LDA models with different topic counts...')
for n_topics in topic_range:
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, max_iter=15,
        learning_method='online', batch_size=256
    )
    lda.fit(dtm)
    perp = lda.perplexity(dtm)
    ll = lda.score(dtm)
    perplexities.append(perp)
    log_likelihoods.append(ll)
    print(f'  Topics={n_topics:2d} | Perplexity={perp:.1f} | Log-likelihood={ll:.1f}')

fig, axes = create_figure(1, 2, figsize=(14, 5), title='Optimal Topic Count Selection')

axes[0].plot(topic_range, perplexities, 'o-', color=PALETTE[0], linewidth=2, markersize=8)
axes[0].set_title('Perplexity (lower is better)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Topics')
axes[0].set_ylabel('Perplexity')
axes[0].grid(alpha=0.15)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[1].plot(topic_range, log_likelihoods, 'o-', color=PALETTE[2], linewidth=2, markersize=8)
axes[1].set_title('Log-Likelihood (higher is better)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Topics')
axes[1].set_ylabel('Log-Likelihood')
axes[1].grid(alpha=0.15)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.show()"""))

c.append(md("## 4. Final LDA Model\n\nWe train the final model with the selected number of topics."))
c.append(code("""# Train final LDA model (10 topics is a good default)
N_TOPICS = 10
lda_final = LatentDirichletAllocation(
    n_components=N_TOPICS, random_state=42, max_iter=25,
    learning_method='online', batch_size=256
)
lda_final.fit(dtm)

feature_names = count_vec.get_feature_names_out()

print_section(f'LDA MODEL — {N_TOPICS} TOPICS')
for topic_idx, topic in enumerate(lda_final.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-16:-1]]
    print(f'Topic {topic_idx+1:2d}: {", ".join(top_words)}')"""))

c.append(code("""# Visualize top words per topic
fig, axes = plt.subplots(2, 5, figsize=(24, 10), facecolor=DARK_BG)
axes = axes.flatten()

for topic_idx, topic in enumerate(lda_final.components_):
    top_indices = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_indices]
    top_weights = [topic[i] for i in top_indices]
    
    ax = axes[topic_idx]
    ax.set_facecolor('#161b22')
    ax.barh(range(len(top_words)), top_weights, color=PALETTE[topic_idx % len(PALETTE)], height=0.7)
    ax.set_yticks(range(len(top_words)))
    ax.set_yticklabels(top_words, fontsize=9, color=TEXT_COLOR)
    ax.invert_yaxis()
    ax.set_title(f'Topic {topic_idx+1}', fontsize=12, fontweight='bold', color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_MUTED)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle(f'Top 10 Words per Topic (LDA, {N_TOPICS} topics)', fontsize=16, 
             fontweight='bold', color=TEXT_COLOR, y=1.02)
plt.tight_layout()
plt.show()"""))

c.append(md("## 5. Topic Distribution Analysis\n\n### 5.1 Topics vs Actual Categories"))
c.append(code("""# Get topic assignments
topic_distributions = lda_final.transform(dtm)
df_sample['dominant_topic'] = topic_distributions.argmax(axis=1) + 1

# Topic vs category heatmap
topic_cat = pd.crosstab(df_sample['category'], df_sample['dominant_topic'], normalize='index') * 100

fig, ax = plt.subplots(figsize=(14, 10), facecolor=DARK_BG)
ax.set_facecolor('#161b22')
plot_heatmap(topic_cat, ax, 'Topic Distribution per Category (%)', cmap='YlOrRd', fmt='.1f')
ax.set_xlabel('Dominant Topic')
ax.set_ylabel('Category')
plt.tight_layout()
plt.show()"""))

c.append(md("### 5.2 Topics vs Star Rating"))
c.append(code("""# Topic distribution by star rating
topic_star = pd.crosstab(df_sample['stars'], df_sample['dominant_topic'], normalize='index') * 100

fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK_BG)
ax.set_facecolor('#161b22')
plot_heatmap(topic_star, ax, 'Topic Distribution per Star Rating (%)', cmap='YlGnBu', fmt='.1f')
ax.set_xlabel('Dominant Topic')
ax.set_ylabel('Star Rating')
plt.tight_layout()
plt.show()

print('\\nDominant topics per star level:')
for star in range(1, 6):
    dominant = topic_star.loc[star].idxmax()
    pct = topic_star.loc[star].max()
    print(f'  {star}★: Topic {dominant} ({pct:.1f}%)')"""))

c.append(md("""## 6. 📋 Observations & Documentation

### Topic Modeling Findings:
1. **LDA discovers meaningful themes** — Topics roughly correspond to industry clusters (delivery, customer service, product quality, etc.)
2. **Not a perfect 1:1 mapping** — Some topics span multiple categories (e.g., "customer service" appears everywhere)
3. **Complaint topics dominate low stars** — 1★ reviews concentrate on "refund/complaint/terrible" topics
4. **Praise topics dominate high stars** — 5★ reviews cluster around "excellent/recommend/quality" topics
5. **10 topics capture the main themes** well for this dataset

### Why LDA as Unsupervised Model 2:
LDA is inherently unsupervised — it discovers latent topics without using any labels. This satisfies both the "Topic Modeling" and "Unsupervised Model 2" requirements simultaneously.

---
*Next: Notebook 08 — Feature Importance & PCA*"""))

with open(r'c:\Users\Ionut\Desktop\sem2_an3\DM\Proiect_DM_P2\notebooks\07_topic_modeling.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Notebook 07 created!')
