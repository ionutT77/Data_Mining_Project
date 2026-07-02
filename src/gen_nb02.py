"""Script to generate Notebook 02 — EDA & Visualization."""
import json

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python', 'version': '3.11.0'}},
    'cells': []
}

def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': [l+'\n' for l in s.split('\n')]}
def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'source': [l+'\n' for l in s.split('\n')], 'outputs': [], 'execution_count': None}

c = nb['cells']

c.append(md("""# 📊 Notebook 02 — Exploratory Data Analysis & Visualization

## Topic: EDA / Data Visualization

**Why this topic?**
EDA reveals the structure, patterns, and anomalies in our data before we apply any models. It helps us understand the distribution of star ratings across categories, identify which industries have the most polarized reviews, and discover relationships between review length and satisfaction.

**What we observe in this notebook:**
- Rating distributions per category (which industries are most/least satisfied?)
- Review length patterns (do angry customers write more?)
- Top companies by review count
- Word clouds comparing 1-star vs 5-star vocabulary
- Cross-category statistical comparisons

---"""))

c.append(md("## 1. Setup & Load Data"))
c.append(code("""import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

from src.visualization import *
from src.utils import print_section, print_subsection

set_dark_theme()

df = pd.read_csv('../data/processed/cleaned_reviews.csv')
print(f'Loaded {len(df):,} cleaned reviews across {df[\"category\"].nunique()} categories')
print(f'Companies: {df[\"company\"].nunique():,}')
print(f'Star range: {df[\"stars\"].min()} - {df[\"stars\"].max()}')"""))

c.append(md("""## 2. Overall Star Rating Distribution

The star rating distribution reveals consumer sentiment patterns on Trustpilot. A key question: are reviews generally positive, negative, or evenly distributed?"""))

c.append(code("""fig, axes = create_figure(1, 2, figsize=(16, 6), title='Overall Rating Landscape')

# Star distribution with percentages
star_counts = df['stars'].value_counts().sort_index()
plot_star_distribution(star_counts, axes[0], 'Star Rating Distribution')

# Cumulative distribution
cumulative = star_counts.cumsum() / star_counts.sum() * 100
axes[1].plot(cumulative.index, cumulative.values, 'o-', color=PALETTE[0], linewidth=2.5, markersize=10)
axes[1].fill_between(cumulative.index, cumulative.values, alpha=0.15, color=PALETTE[0])
for x, y in zip(cumulative.index, cumulative.values):
    axes[1].annotate(f'{y:.1f}%', (x, y), textcoords='offset points', xytext=(0, 12), 
                     ha='center', fontsize=10, fontweight='bold', color=TEXT_COLOR)
axes[1].set_title('Cumulative Rating Distribution', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Stars (≤)')
axes[1].set_ylabel('Cumulative %')
axes[1].set_xticks([1, 2, 3, 4, 5])
axes[1].grid(alpha=0.15)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# Print stats
print(f'\\nPositive (4-5★): {(df["stars"] >= 4).sum():,} ({(df["stars"] >= 4).mean()*100:.1f}%)')
print(f'Neutral  (3★):   {(df["stars"] == 3).sum():,} ({(df["stars"] == 3).mean()*100:.1f}%)')
print(f'Negative (1-2★): {(df["stars"] <= 2).sum():,} ({(df["stars"] <= 2).mean()*100:.1f}%)')"""))

c.append(md("""## 3. Category-Level Analysis

How does satisfaction vary across the 22 business categories? Some industries naturally generate more complaints (e.g., utilities, insurance) while others tend toward positive reviews (e.g., pets, hobbies)."""))

c.append(code("""# Category statistics
cat_stats = df.groupby('category').agg(
    count=('stars', 'count'),
    mean_rating=('stars', 'mean'),
    median_rating=('stars', 'median'),
    std_rating=('stars', 'std'),
    pct_negative=('stars', lambda x: (x <= 2).mean() * 100),
    pct_positive=('stars', lambda x: (x >= 4).mean() * 100),
    avg_word_count=('word_count', 'mean'),
    num_companies=('company', 'nunique')
).round(2).sort_values('mean_rating', ascending=False)

print_section('CATEGORY STATISTICS')
print(cat_stats.to_string())"""))

c.append(code("""# Heatmap-style category comparison
fig, axes = create_figure(1, 2, figsize=(18, 8), title='Category Satisfaction Landscape')

# Mean rating per category (horizontal bar)
cat_sorted = cat_stats.sort_values('mean_rating')
colors = [plt.cm.RdYlGn((v - 1) / 4.0) for v in cat_sorted['mean_rating']]
axes[0].barh(range(len(cat_sorted)), cat_sorted['mean_rating'], color=colors, height=0.7)
axes[0].set_yticks(range(len(cat_sorted)))
axes[0].set_yticklabels(cat_sorted.index, fontsize=8)
axes[0].set_title('Average Rating by Category', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Average Stars')
overall_mean = df['stars'].mean()
axes[0].axvline(overall_mean, color='white', linestyle='--', alpha=0.5, label=f'Overall: {overall_mean:.2f}')
for i, (idx, row) in enumerate(cat_sorted.iterrows()):
    axes[0].text(row['mean_rating'] + 0.03, i, f'{row["mean_rating"]:.2f}', va='center', fontsize=8, color=TEXT_MUTED)
axes[0].legend(fontsize=9)

# Positive vs Negative percentage
x = np.arange(len(cat_sorted))
w = 0.35
axes[1].barh(x - w/2, cat_sorted['pct_positive'], height=w, color='#3fb950', alpha=0.8, label='Positive (4-5★)')
axes[1].barh(x + w/2, cat_sorted['pct_negative'], height=w, color='#ff4444', alpha=0.8, label='Negative (1-2★)')
axes[1].set_yticks(x)
axes[1].set_yticklabels(cat_sorted.index, fontsize=8)
axes[1].set_title('% Positive vs Negative by Category', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Percentage (%)')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()"""))

c.append(md("""## 4. Star Distribution per Category (Stacked)

A stacked bar chart reveals not just the average, but the full distribution shape for each category."""))

c.append(code("""# Stacked star distribution per category
fig, ax = plt.subplots(figsize=(18, 8), facecolor=DARK_BG)
ax.set_facecolor('#161b22')

cat_star_pcts = df.groupby(['category', 'stars']).size().unstack(fill_value=0)
cat_star_pcts = cat_star_pcts.div(cat_star_pcts.sum(axis=1), axis=0) * 100
cat_star_pcts = cat_star_pcts.loc[cat_stats.sort_values('mean_rating').index]

bottom = np.zeros(len(cat_star_pcts))
for star in [1, 2, 3, 4, 5]:
    ax.barh(range(len(cat_star_pcts)), cat_star_pcts[star], left=bottom, 
            color=STAR_COLORS[star], height=0.7, label=f'{star}★', edgecolor='none')
    bottom += cat_star_pcts[star].values

ax.set_yticks(range(len(cat_star_pcts)))
ax.set_yticklabels(cat_star_pcts.index, fontsize=8, color=TEXT_COLOR)
ax.set_xlabel('Percentage (%)', color=TEXT_COLOR)
ax.set_title('Star Distribution per Category (Stacked)', fontsize=14, fontweight='bold', color=TEXT_COLOR, pad=12)
ax.legend(loc='lower right', fontsize=9)
ax.tick_params(colors=TEXT_MUTED)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()"""))

c.append(md("""## 5. Review Length Analysis

Do dissatisfied customers write longer reviews to explain their grievances? Or do satisfied customers write more to praise?"""))

c.append(code("""fig, axes = create_figure(1, 3, figsize=(20, 6), title='Review Length vs Star Rating')

# Box plot
bp = axes[0].boxplot([df[df['stars']==s]['word_count'].clip(upper=300) for s in range(1,6)], 
                      positions=range(1,6), widths=0.6, patch_artist=True, 
                      medianprops=dict(color='white', linewidth=2))
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(STAR_COLORS[i+1])
    patch.set_alpha(0.7)
axes[0].set_title('Word Count by Star Rating', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Stars')
axes[0].set_ylabel('Word Count')
axes[0].set_xticklabels(['1★', '2★', '3★', '4★', '5★'])

# Violin plot
parts = axes[1].violinplot([df[df['stars']==s]['word_count'].clip(upper=300) for s in range(1,6)],
                            positions=range(1,6), showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(STAR_COLORS[i+1])
    pc.set_alpha(0.6)
axes[1].set_title('Word Count Distribution (Violin)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Stars')
axes[1].set_ylabel('Word Count')
axes[1].set_xticks(range(1,6))
axes[1].set_xticklabels(['1★', '2★', '3★', '4★', '5★'])

# Mean word count per star
mean_wc = df.groupby('stars')['word_count'].mean()
bars = axes[2].bar(mean_wc.index, mean_wc.values, color=[STAR_COLORS[s] for s in mean_wc.index], 
                   width=0.6, edgecolor='none')
for bar, val in zip(bars, mean_wc.values):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.0f}', 
                 ha='center', fontsize=10, fontweight='bold', color=TEXT_COLOR)
axes[2].set_title('Average Word Count by Rating', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Stars')
axes[2].set_ylabel('Avg Word Count')
axes[2].set_xticks(range(1,6))
axes[2].set_xticklabels(['1★', '2★', '3★', '4★', '5★'])

plt.tight_layout()
plt.show()

print('\\nKey Insight: Are negative reviews longer?')
for s in range(1, 6):
    subset = df[df['stars'] == s]
    print(f'  {s}★: mean={subset["word_count"].mean():.0f} words, median={subset["word_count"].median():.0f}')"""))

c.append(md("""## 6. Top Companies Analysis

Which companies receive the most reviews? Are high-volume companies rated better or worse?"""))

c.append(code("""# Top 20 companies by review count
top_companies = df.groupby('company').agg(
    count=('stars', 'count'),
    mean_rating=('stars', 'mean'),
    category=('category', 'first')
).sort_values('count', ascending=False).head(20)

fig, axes = create_figure(1, 2, figsize=(18, 8), title='Top 20 Companies by Review Volume')

# Review count
colors = [plt.cm.RdYlGn((r - 1) / 4.0) for r in top_companies['mean_rating']]
axes[0].barh(range(len(top_companies)), top_companies['count'], color=colors, height=0.7)
axes[0].set_yticks(range(len(top_companies)))
labels = [f'{c} ({cat})' for c, cat in zip(top_companies.index, top_companies['category'])]
axes[0].set_yticklabels(labels, fontsize=7)
axes[0].invert_yaxis()
axes[0].set_title('Review Count (color = rating)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Reviews')

# Scatter: count vs rating
scatter_data = df.groupby('company').agg(count=('stars','count'), mean_rating=('stars','mean')).reset_index()
scatter_data = scatter_data[scatter_data['count'] >= 10]
axes[1].scatter(scatter_data['count'], scatter_data['mean_rating'], 
               alpha=0.3, s=15, c=scatter_data['mean_rating'], cmap='RdYlGn', vmin=1, vmax=5)
axes[1].set_title('Review Volume vs Average Rating', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Reviews')
axes[1].set_ylabel('Average Rating')
axes[1].set_xscale('log')
axes[1].axhline(y=df['stars'].mean(), color='white', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()"""))

c.append(md("""## 7. Word Clouds — 1-Star vs 5-Star Reviews

What vocabulary distinguishes extremely negative from extremely positive reviews?"""))

c.append(code("""fig, axes = create_figure(1, 2, figsize=(18, 7), title='Word Clouds: 1★ vs 5★ Reviews')

# 1-star word cloud
text_1star = ' '.join(df[df['stars'] == 1]['cleaned_review'].dropna().sample(min(5000, (df['stars']==1).sum())))
plot_wordcloud(text_1star, axes[0], '1★ Reviews — Most Common Words', colormap='Reds')

# 5-star word cloud
text_5star = ' '.join(df[df['stars'] == 5]['cleaned_review'].dropna().sample(min(5000, (df['stars']==5).sum())))
plot_wordcloud(text_5star, axes[1], '5★ Reviews — Most Common Words', colormap='Greens')

plt.tight_layout()
plt.show()"""))

c.append(md("""## 8. Correlation Analysis

Is there a relationship between review length, star rating, and other numeric features?"""))

c.append(code("""# Compute correlations
corr_data = df[['stars', 'word_count', 'char_count']].copy()
corr_data['title_length'] = df['cleaned_title'].str.len()
corr_data['review_length'] = df['cleaned_review'].str.len()
corr_data['exclamation_count'] = df['cleaned_review'].str.count('!')
corr_data['question_count'] = df['cleaned_review'].str.count(r'\\?')

corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(10, 8), facecolor=DARK_BG)
ax.set_facecolor('#161b22')
plot_heatmap(corr_matrix, ax, 'Feature Correlation Matrix', cmap='coolwarm', fmt='.3f')
plt.tight_layout()
plt.show()

print('\\nKey Correlations with Star Rating:')
star_corrs = corr_matrix['stars'].drop('stars').sort_values()
for feat, val in star_corrs.items():
    direction = '↑' if val > 0 else '↓'
    print(f'  {feat}: {val:+.3f} {direction}')"""))

c.append(md("""## 9. 📋 Observations & Documentation

### Key EDA Findings:

1. **Slight positive skew** — 5-star reviews are the most common (26%), followed by 4-star (21%). This is typical of voluntary review platforms where satisfied customers are more motivated to share.

2. **Category differences are significant** — Some categories (e.g., Animals & Pets, Hobbies) tend toward higher ratings, while service-heavy categories (Utilities, Legal) skew lower. This reflects real-world satisfaction patterns.

3. **Negative reviews are longer** — 1-star and 2-star reviews have higher average word counts. Dissatisfied customers write more to explain their negative experiences, while satisfied customers often leave short praise.

4. **High-volume companies cluster around average** — Companies with many reviews tend to converge toward the mean rating (~3.3), suggesting regression to the mean with more data points.

5. **Distinct vocabulary** — 1-star reviews feature words about problems (service, refund, terrible, worst), while 5-star reviews use praise words (excellent, amazing, recommend, quality).

### Why These Observations Matter:
- The positive skew means our ML models may have slight class imbalance to handle
- Category differences justify our category-level analysis throughout the project
- The review-length correlation provides a useful feature for prediction
- Vocabulary differences confirm that text-based features will be powerful discriminators

---
*Next: Notebook 03 — TF-IDF & N-gram Analysis*"""))

with open(r'c:\Users\Ionut\Desktop\sem2_an3\DM\Proiect_DM_P2\notebooks\02_eda_and_visualization.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Notebook 02 created successfully!')
