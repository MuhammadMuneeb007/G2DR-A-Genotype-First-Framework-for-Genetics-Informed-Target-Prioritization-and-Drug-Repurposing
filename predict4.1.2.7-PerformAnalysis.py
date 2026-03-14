#!/usr/bin/env python3
"""
Nature Genetics 4-Way Venn: Druggable, Hub, Migraine, High Importance
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import matplotlib.patches as mpatches
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

phenotype = sys.argv[1]

# Load data
base = f"/data/ascher02/uqmmune1/ANNOVAR/{phenotype}/GeneDifferentialExpression/Files/Analysis/druggability_analysis/"
df = pd.read_csv(f"{base}{phenotype}_druggability_complete.csv")

# Define gene sets
druggable = set(df[(df['Has_Drugs'] == True) | (df['Druggability_Probability'] >= 0.5)]['Symbol'].dropna())
hub = set(df[df['Hub_Class'].isin(['High_Hub', 'Moderate_Hub'])]['Symbol'].dropna())
migraine = set(df[df['Status'] == 'Existing']['Symbol'].dropna())
high_importance = set(df[df['Importance_Score'] >= df['Importance_Score'].quantile(0.75)]['Symbol'].dropna())

# Nature Genetics style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 0.8

fig = plt.figure(figsize=(10, 5))

# ============================================================================
# PANEL A: 3-Way Venn with Importance Score overlay
# ============================================================================
ax1 = fig.add_subplot(1, 2, 1)

v = venn3([druggable, hub, migraine],
          set_labels=('', '', ''),  # We'll add custom labels
          set_colors=('#2E7D32', '#1565C0', '#F57C00'),
          alpha=0.6,
          ax=ax1)

venn3_circles([druggable, hub, migraine], linewidth=1.2, color='black', ax=ax1)

# Calculate mean importance for each region
def get_region_genes(d, h, m, region):
    """Get genes in specific Venn region"""
    if region == '100': return d - h - m
    if region == '010': return h - d - m
    if region == '001': return m - d - h
    if region == '110': return (d & h) - m
    if region == '101': return (d & m) - h
    if region == '011': return (h & m) - d
    if region == '111': return d & h & m

def mean_importance(genes):
    """Get mean importance score for gene set"""
    if not genes:
        return 0
    scores = df[df['Symbol'].isin(genes)]['Importance_Score'].dropna()
    return scores.mean() if len(scores) > 0 else 0

# Update labels with count and mean importance
regions = ['100', '010', '001', '110', '101', '011', '111']
for region in regions:
    label = v.get_label_by_id(region)
    if label:
        genes = get_region_genes(druggable, hub, migraine, region)
        count = len(genes)
        mean_imp = mean_importance(genes)
        if count > 0:
            label.set_text(f'{count}\n({mean_imp:.2f})')
            label.set_fontsize(9)
            label.set_fontweight('bold')

# Custom set labels with counts
ax1.text(-0.55, 0.3, f'Druggable\n(n={len(druggable)})', fontsize=10, fontweight='bold', 
         ha='center', color='#2E7D32')
ax1.text(0.55, 0.3, f'Hub\n(n={len(hub)})', fontsize=10, fontweight='bold', 
         ha='center', color='#1565C0')
ax1.text(0, -0.55, f'Migraine\n(n={len(migraine)})', fontsize=10, fontweight='bold', 
         ha='center', color='#F57C00')

ax1.set_title('a', fontsize=12, fontweight='bold', loc='left', x=-0.05, y=1.02)
ax1.text(0.5, 1.05, 'Gene Set Overlaps', transform=ax1.transAxes, ha='center', 
         fontsize=11, fontweight='bold')
ax1.text(0.5, -0.18, 'Count (mean importance score)', transform=ax1.transAxes, 
         ha='center', fontsize=8, style='italic')

# ============================================================================
# PANEL B: High Importance intersection with other categories
# ============================================================================
ax2 = fig.add_subplot(1, 2, 2)

# Calculate intersections with high importance
categories = [
    ('Druggable', druggable, '#2E7D32'),
    ('Hub', hub, '#1565C0'),
    ('Migraine', migraine, '#F57C00'),
    ('Drug + Hub', druggable & hub, '#1B5E20'),
    ('Drug + Mig', druggable & migraine, '#E65100'),
    ('Hub + Mig', hub & migraine, '#0D47A1'),
    ('All three', druggable & hub & migraine, '#4A148C')
]

labels = []
total_counts = []
hi_counts = []
colors_list = []

for name, gene_set, color in categories:
    labels.append(name)
    total_counts.append(len(gene_set))
    hi_counts.append(len(gene_set & high_importance))
    colors_list.append(color)

x = np.arange(len(labels))
width = 0.35

bars1 = ax2.bar(x - width/2, total_counts, width, label='Total', color=colors_list, alpha=0.4, edgecolor='black', linewidth=0.5)
bars2 = ax2.bar(x + width/2, hi_counts, width, label='High Importance', color=colors_list, edgecolor='black', linewidth=0.5)

ax2.set_ylabel('Number of genes', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax2.legend(frameon=False, fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add counts
for bar, val in zip(bars2, hi_counts):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_counts)*0.01,
                str(val), ha='center', va='bottom', fontsize=7, fontweight='bold')

ax2.set_title('b', fontsize=12, fontweight='bold', loc='left', x=-0.05, y=1.02)
ax2.text(0.5, 1.05, 'High Importance Genes (top 25%)', transform=ax2.transAxes, 
         ha='center', fontsize=11, fontweight='bold')

# ============================================================================
# Main title
# ============================================================================
fig.suptitle(f'{phenotype.upper()} Drug Repurposing Targets (n={len(df):,})', 
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()

# Save
plt.savefig(f"{base}venn_importance.pdf", dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig(f"{base}venn_importance.png", dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig(f"{base}venn_importance.svg", bbox_inches='tight', facecolor='white')

# ============================================================================
# Print priority targets
# ============================================================================
priority = druggable & hub & high_importance
print(f"\n{'='*60}")
print(f"PRIORITY DRUG REPURPOSING TARGETS")
print(f"{'='*60}")
print(f"Druggable + Hub + High Importance: {len(priority)} genes")
if priority:
    priority_df = df[df['Symbol'].isin(priority)].sort_values('Importance_Score', ascending=False)
    print(priority_df[['Symbol', 'Status', 'Importance_Score', 'Has_Drugs', 
                       'Druggability_Probability', 'Hub_Class']].to_string(index=False))
print(f"{'='*60}")
print(f"\nSaved to {base}venn_importance.[pdf/png/svg]")