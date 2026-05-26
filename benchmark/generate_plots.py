import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Load and preprocess data
df = pd.read_csv('report_v9_score.csv', sep=';',
                 names=['pmid', 'method_id', 'comp_jaccard', 'term_jaccard', 'ted'])

df = df.drop_duplicates(subset=['pmid', 'method_id'])
df['combined'] = (
    0.2 * df['comp_jaccard']
    + 0.5 * df['comp_jaccard'] * df['term_jaccard'] 
    + 0.3 * df['comp_jaccard'] *  df['ted']
)

METHOD_COLORS = {1: '#378ADD', 2: '#1D9E75', 3: '#BA7517', 4: '#D4537E'}
METHOD_LABELS = {1: 'Marker Extraction', 2: 'MinerU Image', 3: 'MinerU Text', 4: 'XML'}
METHODS = [1, 2, 3, 4]
PMID_LABELS = {p: str(int(p))[-5:] for p in df['pmid'].unique()}

# ------- Font sizes  -------------------─────────────────
TITLE_FS   = 14   # subplot / figure title
LABEL_FS   = 13   # axis labels
TICK_FS    = 12   # tick labels
LEGEND_FS  = 11   # legend text


# ------- Plot 1: Box plots  -------------------
fig, axes = plt.subplots(1, 3, figsize=(9, 4))

score_cols = [
    ('comp_jaccard', 'Compartment Jaccard'),
    ('term_jaccard', 'Term-set Jaccard'),
    ('ted',          'Tree-Edit Similarity'),
]

for ax, (col, title) in zip(axes, score_cols):
    data_by_method = [df[df['method_id'] == m][col].dropna().values for m in METHODS]
    bp = ax.boxplot(
        data_by_method,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(linewidth=2.5),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker='o', markersize=5, linestyle='none'),
    )
    for patch, m in zip(bp['boxes'], METHODS):
        patch.set_facecolor(METHOD_COLORS[m] + '55')
        patch.set_edgecolor(METHOD_COLORS[m])
    for median, m in zip(bp['medians'], METHODS):
        median.set_color(METHOD_COLORS[m])
    for whisker, cap, m in zip(
        zip(bp['whiskers'][::2], bp['whiskers'][1::2]),
        zip(bp['caps'][::2],     bp['caps'][1::2]),
        METHODS
    ):
        for line in whisker + cap:
            line.set_color(METHOD_COLORS[m])
    for flier, m in zip(bp['fliers'], METHODS):
        flier.set_markerfacecolor(METHOD_COLORS[m])
        flier.set_markeredgecolor(METHOD_COLORS[m])

    ax.set_xticks([1, 2, 3, 4])

    ax.set_xticklabels(
        [
            'Marker\nHTML',
            'MinerU\nImage',
            'MinerU\nText',
            'XML\nMarkup'
        ],
        fontsize=TICK_FS ,
        rotation=45,
        ha='center',
        linespacing=1.0
    )

    ax.tick_params(axis='x')
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(title, fontsize=TITLE_FS, pad=8)
    ax.set_ylabel('Score', fontsize=LABEL_FS)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

legend_handles = [
    mpatches.Patch(facecolor=METHOD_COLORS[m] + '55',
                   edgecolor=METHOD_COLORS[m], label=METHOD_LABELS[m])
    for m in METHODS
]
# fig.legend(handles=legend_handles, loc='lower center', ncol=4,
#            frameon=False, fontsize=LEGEND_FS, bbox_to_anchor=(0.5, -0.05))
# fig.suptitle('Score distributions by extraction method', fontsize=TITLE_FS + 1, y=1.02)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('plot_1_boxplots.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved plot_1_boxplots.png")


# ------- Plot 2: Scatter -------------------
fig, ax = plt.subplots(figsize=(7, 6))
markers = {1: 'o', 2: '^', 3: 's', 4: 'X'}

for m in METHODS:
    sub = df[df['method_id'] == m]
    ax.scatter(
        sub['comp_jaccard'], sub['term_jaccard'],
        c=METHOD_COLORS[m], marker=markers[m],
        s=70, alpha=0.85, edgecolors='white', linewidths=0.5,
        label=METHOD_LABELS[m], zorder=3,
    )

ax.set_xlabel('Compartment Jaccard', fontsize=LABEL_FS)
ax.set_ylabel('Term-set Jaccard', fontsize=LABEL_FS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.set_xlim(-0.05, 1.1)
ax.set_ylim(-0.05, 1.1)
ax.set_title('Compartment Jaccard vs Term-set Jaccard', fontsize=TITLE_FS)
ax.legend(frameon=False, fontsize=LEGEND_FS)
ax.yaxis.grid(True, linewidth=0.5, alpha=0.5)
ax.xaxis.grid(True, linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('plot_2_scatter.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved plot_2_scatter.png")


# ------- Plot 3: Combined score grouped bar -------------------
pmids  = sorted(df['pmid'].unique())
x      = np.arange(len(pmids))
width  = 0.2

# Thinner width (was 14×5), taller relative to width so bars are prominent
fig, ax = plt.subplots(figsize=(11, 6))

for i, m in enumerate(METHODS):
    vals    = [df[(df['pmid'] == p) & (df['method_id'] == m)]['combined'].values for p in pmids]
    heights = [v[0] if len(v) else np.nan for v in vals]
    ax.bar(x + i * width, heights, width, color=METHOD_COLORS[m],
           label=METHOD_LABELS[m], alpha=0.85)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f"({id+1})" for id,p in enumerate(pmids)],
                   rotation=45, ha='right', fontsize=TICK_FS - 1)
ax.tick_params(axis='y', labelsize=TICK_FS)
ax.set_ylabel('Combined score', fontsize=LABEL_FS)
ax.set_ylim(0, 1.15)
# ax.set_title('Combined score by PMID and extraction method', fontsize=TITLE_FS)
ax.legend(frameon=False, fontsize=LEGEND_FS, ncol=4)
ax.yaxis.grid(True, linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('plot_3_combined_bar.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved plot_3_combined_bar.png")


# ------- Plots 4 & 5: Heatmaps -------------------
pivot = df.pivot(index='pmid', columns='method_id', values='combined')
pivot.index   = [PMID_LABELS[p] for p in pivot.index]
pivot.columns = [f'M{m}' for m in pivot.columns]

mean_order = pivot.mean(axis=1).sort_values().index
var_order  = pivot.var(axis=1).sort_values(ascending=False).index

cmap = sns.color_palette("YlGn", as_cmap=True)

for sort_order, label, fname in [
    (mean_order, 'sorted by mean combined score',     'plot_4_heatmap_mean.png'),
    (var_order,  'sorted by variance across methods', 'plot_5_heatmap_variance.png'),
]:
    fig, ax = plt.subplots(figsize=(6, 7))
    sns.heatmap(
        pivot.loc[sort_order],
        ax=ax,
        cmap=cmap,
        vmin=0, vmax=1,
        annot=True, fmt='.2f',
        annot_kws={'size': 10},
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.6, 'label': 'Combined score'},
    )
    ax.set_title(f'Combined score heatmap\n{label}', fontsize=11, pad=10)
    ax.set_xlabel('Method', fontsize=10)
    ax.set_ylabel('PMID (last 5 digits)', fontsize=10)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")