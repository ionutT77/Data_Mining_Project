"""
visualization.py — Reusable Plotting Functions
================================================
Premium, publication-quality charts with a consistent dark theme.
Used across all notebooks for visual consistency.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple


# ── Global Style Configuration ──────────────────────────────────────────────
DARK_BG = '#0d1117'
DARK_CARD = '#161b22'
DARK_BORDER = '#30363d'
TEXT_COLOR = '#e6edf3'
TEXT_MUTED = '#8b949e'
GRID_COLOR = '#21262d'

# Premium color palette (carefully curated for dark backgrounds)
PALETTE = [
    '#58a6ff',  # Blue
    '#f78166',  # Orange/Coral
    '#3fb950',  # Green
    '#d2a8ff',  # Purple
    '#f0e68c',  # Soft Yellow
    '#ff7b72',  # Red
    '#79c0ff',  # Light Blue
    '#ffa657',  # Amber
    '#7ee787',  # Light Green
    '#bc8cff',  # Lavender
    '#ff9bce',  # Pink
    '#56d4dd',  # Teal
]

# Star rating colors (1=red → 5=green)
STAR_COLORS = {
    1: '#ff4444',
    2: '#ff8c42',
    3: '#ffd700',
    4: '#90ee90',
    5: '#3fb950',
}

STAR_PALETTE = [STAR_COLORS[i] for i in range(1, 6)]


def set_dark_theme():
    """Apply the premium dark theme globally."""
    plt.rcParams.update({
        'figure.facecolor': DARK_BG,
        'axes.facecolor': DARK_CARD,
        'axes.edgecolor': DARK_BORDER,
        'axes.labelcolor': TEXT_COLOR,
        'text.color': TEXT_COLOR,
        'xtick.color': TEXT_MUTED,
        'ytick.color': TEXT_MUTED,
        'grid.color': GRID_COLOR,
        'grid.alpha': 0.3,
        'legend.facecolor': DARK_CARD,
        'legend.edgecolor': DARK_BORDER,
        'legend.labelcolor': TEXT_COLOR,
        'figure.dpi': 120,
        'savefig.dpi': 150,
        'savefig.facecolor': DARK_BG,
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
    })


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    suptitle_y: float = 1.02
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a figure with the dark theme applied."""
    set_dark_theme()
    
    if figsize is None:
        figsize = (7 * ncols, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if title:
        fig.suptitle(title, fontsize=18, fontweight='bold', color=TEXT_COLOR, y=suptitle_y)
    
    return fig, axes


def plot_bar(
    data: pd.Series,
    ax: plt.Axes,
    title: str,
    xlabel: str = '',
    ylabel: str = '',
    color: Optional[str] = None,
    horizontal: bool = False,
    top_n: Optional[int] = None,
    annotate: bool = True,
    palette: Optional[list] = None
):
    """Plot a styled bar chart."""
    if top_n:
        data = data.head(top_n)
    
    if palette:
        colors = palette[:len(data)]
    elif color:
        colors = color
    else:
        colors = PALETTE[:len(data)]
    
    if horizontal:
        bars = ax.barh(range(len(data)), data.values, color=colors, edgecolor='none', height=0.7)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index, fontsize=9)
        ax.invert_yaxis()
        if annotate:
            for bar, val in zip(bars, data.values):
                ax.text(bar.get_width() + max(data.values) * 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:,.0f}' if isinstance(val, (int, float, np.integer, np.floating)) else str(val),
                        va='center', fontsize=8, color=TEXT_MUTED)
    else:
        bars = ax.bar(range(len(data)), data.values, color=colors, edgecolor='none', width=0.7)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=9)
        if annotate:
            for bar, val in zip(bars, data.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(data.values) * 0.01,
                        f'{val:,.0f}' if isinstance(val, (int, float, np.integer, np.floating)) else str(val),
                        ha='center', va='bottom', fontsize=8, color=TEXT_MUTED)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis='x' if horizontal else 'y', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_distribution(
    data: pd.Series,
    ax: plt.Axes,
    title: str,
    xlabel: str = '',
    color: str = '#58a6ff',
    bins: int = 50,
    kde: bool = True
):
    """Plot a styled histogram/distribution."""
    ax.hist(data.dropna(), bins=bins, color=color, alpha=0.7, edgecolor='none', density=kde)
    if kde:
        from scipy import stats
        x_range = np.linspace(data.min(), data.max(), 200)
        kde_vals = stats.gaussian_kde(data.dropna())(x_range)
        ax.plot(x_range, kde_vals, color='#f78166', linewidth=2, alpha=0.8)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Density' if kde else 'Count', fontsize=10)
    ax.grid(axis='y', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_heatmap(
    data: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    cmap: str = 'RdYlGn',
    annot: bool = True,
    fmt: str = '.2f',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
):
    """Plot a styled heatmap."""
    sns.heatmap(
        data, ax=ax, cmap=cmap, annot=annot, fmt=fmt,
        linewidths=0.5, linecolor=DARK_BORDER,
        cbar_kws={'shrink': 0.8},
        vmin=vmin, vmax=vmax,
        annot_kws={'fontsize': 9, 'color': TEXT_COLOR}
    )
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(axis='both', labelsize=9)


def plot_star_distribution(
    star_counts: pd.Series,
    ax: plt.Axes,
    title: str = 'Star Rating Distribution'
):
    """Plot star rating distribution with gradient colors."""
    colors = [STAR_COLORS.get(i, '#58a6ff') for i in star_counts.index]
    bars = ax.bar(star_counts.index, star_counts.values, color=colors, edgecolor='none', width=0.6)
    
    for bar, val in zip(bars, star_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(star_counts.values) * 0.02,
                f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold', color=TEXT_COLOR)
    
    total = star_counts.sum()
    for bar, val in zip(bars, star_counts.values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                f'{pct:.1f}%', ha='center', va='center', fontsize=9, color=DARK_BG, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Stars', fontsize=11)
    ax.set_ylabel('Number of Reviews', fontsize=11)
    ax.set_xticks(star_counts.index)
    ax.set_xticklabels([f'{"★" * i}' for i in star_counts.index], fontsize=12)
    ax.grid(axis='y', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_wordcloud(
    text: str,
    ax: plt.Axes,
    title: str,
    max_words: int = 100,
    colormap: str = 'cool'
):
    """Generate and plot a word cloud."""
    from wordcloud import WordCloud
    
    wc = WordCloud(
        width=800, height=400,
        max_words=max_words,
        background_color=DARK_BG,
        colormap=colormap,
        contour_width=0,
        prefer_horizontal=0.7,
        min_font_size=8,
        max_font_size=80,
    ).generate(text)
    
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.axis('off')


def add_watermark(fig: plt.Figure, text: str = "Trustpilot Reviews Analysis | Data Mining Project 2025"):
    """Add a subtle watermark/footer to the figure."""
    fig.text(0.5, -0.01, text, ha='center', fontsize=8, color=TEXT_MUTED, style='italic')


def save_figure(fig: plt.Figure, name: str, output_dir: str = '../data/figures'):
    """Save figure to disk."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'{name}.png')
    fig.savefig(filepath, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Saved: {filepath}")
