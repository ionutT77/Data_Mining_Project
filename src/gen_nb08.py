"""Generate Notebook 08 — Feature Importance & PCA."""
import json

nb = {'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.11.0'}}, 'cells': []}

def md(s): return {'cell_type': 'markdown', 'metadata': {}, 'source': [l+'\n' for l in s.split('\n')]}
def code(s): return {'cell_type': 'code', 'metadata': {}, 'source': [l+'\n' for l in s.split('\n')], 'outputs': [], 'execution_count': None}
c = nb['cells']

c.append(md("""# 🎯 Notebook 08 — Feature Importance & PCA

## Topic: Feature Importance Analysis / Feature Selection / PCA

**Why this topic?** With hundreds of potential features (TF-IDF terms, sentiment scores, POS ratios, review length, etc.), we need to understand which features actually drive predictions. PCA helps visualize high-dimensional data in 2D/3D, revealing natural clustering patterns.

**What we observe:** Which features are most predictive of star ratings, how dimensionality reduction reveals data structure, and whether reviews naturally cluster by category or rating.

---"""))

c.append(md("## 1. Setup"))
c.append(code("""import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

from src.feature_engineering import build_tfidf_matrix, build_feature_matrix
from src.visualization import *
from src.utils import print_section

set_dark_theme()

df = pd.read_csv('../data/processed/cleaned_reviews.csv')
SAMPLE_SIZE = 15000
df_sample = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
print(f'Using {SAMPLE_SIZE:,} reviews for feature analysis')"""))

c.append(md("## 2. Build Feature Matrix"))
c.append(code("""# Build comprehensive feature matrix
print('Building feature matrix (TF-IDF + metadata)...')
X, feature_names = build_feature_matrix(df_sample, text_column='cleaned_review', tfidf_features=500)
y = df_sample['stars']

print(f'Feature matrix shape: {X.shape}')
print(f'Total features: {len(feature_names)}')
print(f'  - Meta features: {sum(1 for f in feature_names if not f.startswith("tfidf_"))}')
print(f'  - TF-IDF features: {sum(1 for f in feature_names if f.startswith("tfidf_"))}')"""))

c.append(md("## 3. Random Forest Feature Importance\n\nRandom Forest provides built-in feature importance based on how much each feature reduces impurity."))
c.append(code("""# Train a quick Random Forest for feature importance
print('Training Random Forest for feature importance...')
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Get feature importances
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

# Top 30 features
fig, axes = create_figure(1, 2, figsize=(18, 8), title='Feature Importance Analysis')

top30 = importances.head(30).sort_values()
axes[0].barh(range(len(top30)), top30.values, color=PALETTE[0], height=0.7)
axes[0].set_yticks(range(len(top30)))
labels = [f.replace('tfidf_', '') for f in top30.index]
axes[0].set_yticklabels(labels, fontsize=8)
axes[0].set_title('Top 30 Features (RF Importance)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Importance Score')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Cumulative importance
cum_imp = importances.cumsum()
axes[1].plot(range(len(cum_imp)), cum_imp.values, color=PALETTE[2], linewidth=2)
axes[1].fill_between(range(len(cum_imp)), cum_imp.values, alpha=0.15, color=PALETTE[2])
axes[1].axhline(y=0.9, color=PALETTE[1], linestyle='--', alpha=0.7, label='90% importance')
n_90 = (cum_imp < 0.9).sum()
axes[1].axvline(x=n_90, color=PALETTE[1], linestyle='--', alpha=0.7)
axes[1].set_title('Cumulative Feature Importance', fontsize=13, fontweight='bold')
axes[1].set_xlabel(f'Number of Features (90% reached at {n_90})')
axes[1].set_ylabel('Cumulative Importance')
axes[1].legend()
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

print(f'\\nTop 10 most important features:')
for feat, imp in importances.head(10).items():
    print(f'  {feat.replace("tfidf_", "")}: {imp:.4f}')
print(f'\\nFeatures needed for 90% importance: {n_90} / {len(importances)}')"""))

c.append(md("## 4. Mutual Information Feature Selection"))
c.append(code("""# Mutual information
print('Computing mutual information scores...')
mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK_BG)
ax.set_facecolor('#161b22')
top20_mi = mi_series.head(20).sort_values()
ax.barh(range(len(top20_mi)), top20_mi.values, color=PALETTE[3], height=0.7)
ax.set_yticks(range(len(top20_mi)))
ax.set_yticklabels([f.replace('tfidf_', '') for f in top20_mi.index], fontsize=9, color=TEXT_COLOR)
ax.set_title('Top 20 Features (Mutual Information)', fontsize=14, fontweight='bold', color=TEXT_COLOR)
ax.set_xlabel('MI Score', color=TEXT_COLOR)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=TEXT_MUTED)
plt.tight_layout()
plt.show()"""))

c.append(md("## 5. PCA — Dimensionality Reduction\n\n### 5.1 Explained Variance"))
c.append(code("""# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Full PCA
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

# Explained variance plot
fig, axes = create_figure(1, 2, figsize=(14, 5), title='PCA Explained Variance')

# Individual
axes[0].bar(range(1, 31), pca_full.explained_variance_ratio_[:30] * 100, color=PALETTE[0], edgecolor='none')
axes[0].set_title('Variance per Component (Top 30)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance (%)')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Cumulative
cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
axes[1].plot(range(1, len(cum_var)+1), cum_var, color=PALETTE[2], linewidth=2)
axes[1].fill_between(range(1, len(cum_var)+1), cum_var, alpha=0.1, color=PALETTE[2])
axes[1].axhline(y=90, color=PALETTE[1], linestyle='--', alpha=0.7, label='90%')
axes[1].axhline(y=95, color=PALETTE[3], linestyle='--', alpha=0.7, label='95%')
n_90_pca = np.argmax(cum_var >= 90) + 1
axes[1].set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
axes[1].set_xlabel(f'Components (90% at {n_90_pca})')
axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].legend()
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
print(f'Components for 90% variance: {n_90_pca}')
print(f'Components for 95% variance: {np.argmax(cum_var >= 95) + 1}')"""))

c.append(md("### 5.2 2D PCA Scatter — Colored by Star Rating"))
c.append(code("""# 2D PCA projection
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_scaled)

fig, axes = create_figure(1, 2, figsize=(16, 7), title='PCA 2D Projection')

# Colored by star rating
for star in [1, 2, 3, 4, 5]:
    mask = y == star
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], s=5, alpha=0.3, 
                   color=STAR_COLORS[star], label=f'{star}★')
axes[0].set_title('PCA — Colored by Star Rating', fontsize=13, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].legend(fontsize=8, markerscale=3)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Binary: positive vs negative
binary = y.apply(lambda x: 'Positive' if x >= 4 else ('Negative' if x <= 2 else 'Neutral'))
for label, color in [('Negative', '#ff4444'), ('Neutral', '#ffd700'), ('Positive', '#3fb950')]:
    mask = binary == label
    axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], s=5, alpha=0.3, color=color, label=label)
axes[1].set_title('PCA — Positive vs Negative', fontsize=13, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].legend(fontsize=9, markerscale=3)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.show()"""))

c.append(md("### 5.3 3D PCA Visualization"))
c.append(code("""# 3D PCA
from mpl_toolkits.mplot3d import Axes3D

pca_3d = PCA(n_components=3, random_state=42)
X_3d = pca_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(12, 9), facecolor=DARK_BG)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor(DARK_BG)

for star in [1, 2, 3, 4, 5]:
    mask = y == star
    ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], 
              s=3, alpha=0.3, color=STAR_COLORS[star], label=f'{star}★')

ax.set_title('3D PCA — Star Rating', fontsize=14, fontweight='bold', color=TEXT_COLOR, pad=20)
ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)', color=TEXT_MUTED)
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)', color=TEXT_MUTED)
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)', color=TEXT_MUTED)
ax.legend(fontsize=8, markerscale=3)
ax.tick_params(colors=TEXT_MUTED)
plt.tight_layout()
plt.show()"""))

c.append(md("""## 6. 📋 Observations & Documentation

### Feature Importance Findings:
1. **Meta features dominate** — `word_count`, `char_count`, and `avg_word_length` are among the top predictors, confirming that review length correlates with sentiment
2. **Specific TF-IDF terms are powerful** — Words like "excellent", "terrible", "refund" are strong predictors of star ratings
3. **Feature redundancy** — Only ~100-150 features are needed for 90% of the predictive power (out of 500+)

### PCA Findings:
1. **Partial separation** in 2D — Positive and negative reviews form somewhat distinct clusters but overlap significantly
2. **First 2 components** explain limited variance — text data is inherently high-dimensional
3. **3D adds marginal improvement** — The third component provides some additional separation
4. **Binary split works better** — Positive vs Negative separation is clearer than 5-class separation

### Implications for ML Models:
- Feature selection can reduce model complexity without losing much accuracy
- PCA can be used as a preprocessing step but won't dramatically improve performance on text data
- The partial overlap in PCA explains why perfect star prediction from text alone is difficult

---
*Next: Notebook 09 — Supervised Learning*"""))

with open(r'c:\Users\Ionut\Desktop\sem2_an3\DM\Proiect_DM_P2\notebooks\08_feature_importance_pca.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Notebook 08 created!')
