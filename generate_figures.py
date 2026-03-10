#!/usr/bin/env python3
"""
Generate all figures for:
"Who Am I, and Who Else Is Here?" Emergent Behaviors in LLM Group Conversations

Usage:
    python generate_figures.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]

Defaults:
    --data-dir  ./data
    --output-dir ./figures

Required CSV files in data-dir:
    - coded_v4_agreed.csv
    - coded_v4_gemini.csv
    - coded_v4_sonnet.csv
"""

import os
import sys
import csv
import argparse
import numpy as np
from collections import Counter
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

TRAIT_FLAGS = ['is_phatic', 'is_meta', 'is_lead', 'is_arch', 'is_agree']
TRAIT_LABELS = ['Phatic', 'Meta', 'Lead', 'Arch', 'Agree']
ALL_FLAGS = ['is_phatic', 'is_meta', 'is_lead', 'is_arch', 'is_agree', 'is_comp']
ALL_LABELS = ['Phatic', 'Meta', 'Lead', 'Arch', 'Agree', 'Comp']
SENS_FLAGS = ['is_phatic', 'is_meta', 'is_lead', 'is_arch', 'is_agree', 'is_comp', 'has_xref']
SENS_LABELS = ['Phatic', 'Meta', 'Lead', 'Arch', 'Agree', 'Comp', 'Xref']

AGENTS_FULL = [
    'LLaMA 3.3 70B', 'GPT-OSS 120B', 'GPT-OSS 20B',
    'LLaMA 4 Maverick', 'Kimi K2', 'Qwen3 32B', 'LLaMA 4 Scout'
]
AGENTS_SHORT = [
    'LLaMA-70B', 'GPT-120B', 'GPT-20B',
    'Maverick', 'Kimi', 'Qwen-32B', 'Scout'
]

# Colors
C_PLAT   = '#2B6CB0'
C_AGENT  = '#48874A'
C_JUDGE  = '#D4652F'
C_HUMAN  = '#7B2D8E'
C_DATA   = '#5A6672'
C_OUTPUT = '#C0392B'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def load_csv_dict(path, key='id'):
    with open(path, newline='', encoding='utf-8') as f:
        return {r[key]: r for r in csv.DictReader(f)}


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def profile(agr, series, agent):
    """Compute behavioral profile (5-d vector) for an agent in a series."""
    rows = [r for r in agr if r['series'] == series and r['agent'] == agent]
    if not rows:
        return [0.0] * 5
    n = len(rows)
    return [sum(1 for r in rows if r[f] == 'True') / n for f in TRAIT_FLAGS]


def cosine_per_run(agr, series):
    """Compute mean pairwise cosine similarity per run in a series."""
    srows = [r for r in agr if r['series'] == series]
    runs = sorted(set(r['file'] for r in srows))
    result = []
    for run in runs:
        rr = [r for r in srows if r['file'] == run]
        agents = sorted(set(r['agent'] for r in rr))
        if len(agents) < 2:
            continue
        profiles = {}
        for ag in agents:
            ar = [r for r in rr if r['agent'] == ag]
            n = len(ar)
            if n == 0:
                continue
            profiles[ag] = np.array([
                sum(1 for r in ar if r[f] == 'True') / n for f in TRAIT_FLAGS
            ])
        cosines = []
        for a1, a2 in combinations(profiles, 2):
            v1, v2 = profiles[a1], profiles[a2]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cosines.append(np.dot(v1, v2) / (n1 * n2))
        if cosines:
            result.append(np.mean(cosines))
    return result


def label_along_arrow(ax, x1, y1, x2, y2, text, offset=0.15, fs=8.5, side='right'):
    """Place a label parallel to an arrow, offset to one side."""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    if side == 'right':
        nx = dy / length * offset
        ny = -dx / length * offset
    else:
        nx = -dy / length * offset
        ny = dx / length * offset
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < -90:
        angle += 180
    elif angle > 90:
        angle -= 180
    ax.text(mx + nx, my + ny, text, ha='center', va='center', fontsize=fs,
            color='#666666', style='italic', family='sans-serif', zorder=5,
            rotation=angle, rotation_mode='anchor')


# ═══════════════════════════════════════════════════════════════
# FIGURE 0 — Experimental Pipeline
# ═══════════════════════════════════════════════════════════════

def fig0_pipeline(out):
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.5); ax.axis('off')

    def box(cx, cy, w, h, color, text, fs=11):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.12", facecolor=color,
            edgecolor='none', zorder=3, alpha=0.95)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
                color='white', fontweight='bold', zorder=4,
                linespacing=1.3, family='sans-serif')

    def arr(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.8,
                                    shrinkA=3, shrinkB=3), zorder=2)

    ax.text(5, 10.1, 'Experimental Pipeline', ha='center', fontsize=18,
            fontweight='bold', family='sans-serif', color='#1a1a1a')

    # Row 1
    y1 = 9.2
    box(1.0, y1, 1.7, 0.8, C_PLAT, 'War Room\n(Orchestrator)', 10)
    box(3.5, y1, 1.4, 0.8, C_PLAT, 'Groq\n(inference)', 10)
    box(5.6, y1, 1.6, 0.8, C_AGENT, '7 LLMs\n× 208 runs', 10)
    box(8.2, y1, 1.7, 0.8, C_DATA, '13,786\nmessages', 11)

    arr(1.9, y1, 2.75, y1)
    ax.text(2.32, y1 + 0.38, 'API calls', ha='center', va='bottom', fontsize=9,
            color='#444', fontweight='bold', style='italic', family='sans-serif')
    arr(4.25, y1, 4.75, y1)
    arr(6.45, y1, 7.3, y1)
    ax.text(6.88, y1 + 0.38, 'JSON', ha='center', va='bottom', fontsize=9,
            color='#444', fontweight='bold', style='italic', family='sans-serif')

    # 3 branches down
    arr(7.5, 8.78, 2.8, 7.65)
    arr(8.0, 8.78, 5.8, 7.65)
    arr(8.7, 8.78, 8.7, 7.65)
    label_along_arrow(ax, 8.7, 8.78, 8.7, 7.65, 'sample', 0.28, side='right')

    # Row 2
    y2 = 7.2
    box(2.8, y2, 2.3, 0.8, C_JUDGE, 'Gemini 3.1 Pro\n(Judge 1)', 10)
    box(5.8, y2, 2.3, 0.8, C_JUDGE, 'Claude Sonnet 4.6\n(Judge 2)', 10)
    box(8.7, y2, 1.5, 0.8, C_HUMAN, 'Human\n(609 msgs)', 9.5)

    # Judges → Intersection
    arr(2.8, 6.78, 4.2, 5.85)
    arr(5.8, 6.78, 4.8, 5.85)
    box(4.5, 5.4, 2.3, 0.8, C_DATA, 'Intersection\n(both agree)', 10)

    # Human → Broadcast Filter
    arr(8.7, 6.78, 6.8, 4.05)
    label_along_arrow(ax, 8.7, 6.78, 6.8, 4.05, 'identified FP', 0.30, side='left')

    # Intersection → Broadcast Filter
    arr(4.5, 4.98, 5.8, 4.05)
    box(5.8, 3.6, 2.6, 0.8, C_HUMAN, 'Broadcast Filter\n(clean_comp.py)', 10)

    # Output
    arr(5.8, 3.18, 5.8, 2.25)
    box(5.8, 1.8, 1.6, 0.8, C_DATA, 'Clean\nData', 10)
    box(2.8, 1.8, 2.2, 0.8, C_OUTPUT, '5 RQs\n11 Figures', 10)
    arr(5.0, 1.8, 3.95, 1.8)

    # Legend
    items = [('Platform', C_PLAT), ('Agents', C_AGENT), ('LLM judges', C_JUDGE),
             ('Human', C_HUMAN), ('Data', C_DATA), ('Output', C_OUTPUT)]
    sx = 5 - len(items) * 1.55 / 2
    for i, (l, c) in enumerate(items):
        x = sx + i * 1.55
        ax.add_patch(plt.Rectangle((x, 0.42), 0.22, 0.16, facecolor=c, edgecolor='none'))
        ax.text(x + 0.32, 0.5, l, fontsize=8, va='center', color='#555', family='sans-serif')

    fig.savefig(out, facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# FIGURE 1 — Behavioral Profiles Heatmap (Series A)
# ═══════════════════════════════════════════════════════════════

def fig1_profiles(agr, out):
    data = np.array([profile(agr, 'A', ag) for ag in AGENTS_FULL])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)
    ax.set_xticks(range(5)); ax.set_xticklabels(TRAIT_LABELS, fontsize=11, fontweight='bold')
    ax.set_yticks(range(7)); ax.set_yticklabels(AGENTS_SHORT, fontsize=10)
    ax.set_title('Agent Behavioral Profiles — Series A (Baseline)',
                 fontsize=13, fontweight='bold', pad=10)
    for i in range(7):
        for j in range(5):
            v = data[i, j]
            c = 'white' if v > 0.25 else 'black'
            w = 'bold' if v >= 0.20 else 'normal'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color=c, fontsize=11, fontweight=w)
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02).set_label('Proportion')
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# FIGURE 2 — Isolation (F) vs Group (A)
# ═══════════════════════════════════════════════════════════════

def fig2_isolation(agr, out):
    data_f = np.array([profile(agr, 'F', ag) for ag in AGENTS_FULL])
    data_a = np.array([profile(agr, 'A', ag) for ag in AGENTS_FULL])

    fig = plt.figure(figsize=(11, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    im1 = ax1.imshow(data_f, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)
    ax1.set_xticks(range(5)); ax1.set_xticklabels(TRAIT_LABELS)
    ax1.set_yticks(range(7)); ax1.set_yticklabels(AGENTS_SHORT)
    ax1.set_title('F (Isolated) — 4.2% flagged', fontweight='bold')
    for i in range(7):
        for j in range(5):
            ax1.text(j, i, f'{data_f[i, j]:.2f}', ha='center', va='center',
                     color='#555', fontsize=10)

    im2 = ax2.imshow(data_a, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)
    ax2.set_xticks(range(5)); ax2.set_xticklabels(TRAIT_LABELS)
    ax2.set_yticks(range(7)); ax2.set_yticklabels([])
    ax2.set_title('A (Group) — 64.3% flagged', fontweight='bold')
    for i in range(7):
        for j in range(5):
            v = data_a[i, j]
            c = 'white' if v > 0.25 else 'black'
            w = 'bold' if v >= 0.20 else 'normal'
            ax2.text(j, i, f'{v:.2f}', ha='center', va='center',
                     color=c, fontsize=10, fontweight=w)

    fig.colorbar(im2, cax=cax, label='Proportion')
    fig.suptitle('Behavioral Emergence: Isolation vs. Group Context',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.savefig(out, facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# FIGURE 3 — Compensation (RQ2)
# ═══════════════════════════════════════════════════════════════

def fig3_compensation(agr, out):
    a_comp = [r for r in agr if r['series'] == 'A' and r['is_comp'] == 'True']
    ag_comp = Counter(r['agent'] for r in a_comp)
    lv_comp = Counter(r['comp_level'] for r in a_comp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    fig.suptitle('RQ2 — Compensatory Responses After Agent Crash (Series A)',
                 fontsize=12, fontweight='bold', y=1.0)

    short_map = {
        'GPT-OSS 20B': 'GPT-20B', 'GPT-OSS 120B': 'GPT-120B',
        'LLaMA 4 Scout': 'Scout', 'Qwen3 32B': 'Qwen-32B',
        'LLaMA 4 Maverick': 'Maverick'
    }
    ca = ['GPT-20B', 'GPT-120B', 'Scout', 'Qwen-32B', 'Maverick']
    caf = {v: k for k, v in short_map.items()}
    cv = [ag_comp.get(caf.get(a, ''), 0) for a in ca]

    ax1.barh(ca, cv, color=['#5B9BD5'] * 3 + ['#ED7D31'] * 2,
             edgecolor='white', height=0.6)
    ax1.set_xlabel('Compensation messages')
    ax1.set_title('Who compensates?', fontweight='bold')
    for i, v in enumerate(cv):
        if v > 0:
            ax1.text(v + 0.15, i, str(v), va='center', fontweight='bold')

    ln = ['L1\nMention', 'L2\nTakeover', 'L3\nRedistrib.']
    lv = [lv_comp.get('L1', 0), lv_comp.get('L2', 0), lv_comp.get('L3', 0)]
    lc = ['#FFC000', '#ED7D31', '#C00000']
    bars = ax2.bar(ln, lv, color=lc, edgecolor='white', width=0.55)
    ax2.set_ylabel('Count')
    ax2.set_title('Compensation depth', fontweight='bold')
    ax2.set_ylim(0, max(lv) + 2)
    for b, v in zip(bars, lv):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v),
                 ha='center', fontweight='bold', fontsize=12)
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# FIGURE 4 — Name Bias (RQ3)
# ═══════════════════════════════════════════════════════════════

def fig4_name_bias(agr, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle('RQ3 — Name Bias: Neutral vs Real Model Names',
                 fontsize=12, fontweight='bold', y=1.0)

    for ag in AGENTS_FULL:
        pa = profile(agr, 'A', ag)
        pc = profile(agr, 'C', ag)
        delta = [c - a for a, c in zip(pa, pc)]
        ax1.plot(TRAIT_LABELS, delta, 'o-', alpha=0.6, markersize=4, lw=1.5)
    ax1.axhline(0, color='gray', ls='--', alpha=0.4)
    ax1.set_ylabel('$\\Delta$ (C $-$ A)')
    ax1.set_title('Behavioral shift with real names', fontweight='bold')

    cA = np.mean(cosine_per_run(agr, 'A'))
    cC = np.mean(cosine_per_run(agr, 'C'))
    bars = ax2.bar(['A\n(Neutral)', 'C\n(Real names)'], [cA, cC],
                   color=['#5B9BD5', '#ED7D31'], width=0.45, edgecolor='white')
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('Cosine similarity')
    ax2.set_title('Profile similarity', fontweight='bold')
    for b, v in zip(bars, [cA, cC]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02,
                 f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# FIGURE 5 — Heterogeneous vs Homogeneous (RQ4)
# ═══════════════════════════════════════════════════════════════

def fig5_heterogeneity(agr, out):
    cA = cosine_per_run(agr, 'A')
    cB = cosine_per_run(agr, 'B')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle('RQ4 — Heterogeneous vs Homogeneous',
                 fontsize=12, fontweight='bold', y=1.0)

    bp = ax1.boxplot([cA, cB],
                     tick_labels=['A\n(Heterogeneous)', 'B\n(Homogeneous)'],
                     widths=0.35, patch_artist=True, showfliers=True,
                     flierprops=dict(marker='o', markersize=5))
    bp['boxes'][0].set(facecolor='#5B9BD5', alpha=0.7)
    bp['boxes'][1].set(facecolor='#ED7D31', alpha=0.7)
    ax1.set_ylabel('Mean pairwise cosine')
    ax1.set_title('Per-run profile similarity\np < 0.001 ***',
                  fontweight='bold', fontsize=10)

    xr = np.linspace(0.1, 1.0, 300)
    kA = gaussian_kde(cA)(xr)
    kB = gaussian_kde(cB)(xr)
    ax2.fill_between(xr, kA, alpha=0.35, color='#5B9BD5',
                     label=f'A ($\\mu$={np.mean(cA):.2f})')
    ax2.fill_between(xr, kB, alpha=0.35, color='#ED7D31',
                     label=f'B ($\\mu$={np.mean(cB):.2f})')
    ax2.set_xlabel('Pairwise cosine similarity')
    ax2.set_ylabel('Density')
    ax2.set_title('Profile similarity distribution', fontweight='bold')
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# FIGURE 6 — Prompt Ablation (RQ5)
# ═══════════════════════════════════════════════════════════════

def fig6_ablation(agr, out):
    series = ['A', 'K1', 'K2', 'K3']
    labels = ['A\n(Full)', 'K1\n(No peers)', 'K2\n(No ID)', 'K3\n(Empty)']
    means = [np.mean(cosine_per_run(agr, s)) for s in series]
    clone_mean = np.mean(cosine_per_run(agr, 'B'))

    colors = ['#5B9BD5', '#70AD47', '#FFC000', '#ED7D31']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, color=colors, width=0.5,
                  edgecolor='white', zorder=3)
    ax.axhline(clone_mean, color='#C00000', ls='--', lw=2,
               label='B (homogeneous)', zorder=2)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f'{v:.2f}', ha='center', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cosine similarity\n($\\uparrow$ = less differentiated)')
    ax.set_ylim(0, 1.05)
    ax.set_title('RQ5 — Prompt Ablation Gradient',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.2, zorder=0)
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# FIGURE 7 — Confusion Matrices (Gemini × Sonnet)
# ═══════════════════════════════════════════════════════════════

def fig7_confusion(gem, son, out):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    fig.suptitle('Inter-Rater Confusion Matrices — Gemini × Sonnet',
                 fontsize=13, fontweight='bold', y=1.0)

    for idx, (flag, name) in enumerate(zip(ALL_FLAGS, ALL_LABELS)):
        ax = axes[idx // 3][idx % 3]
        a, b, c, d = 0, 0, 0, 0
        for mid in gem:
            g = gem[mid][flag] == 'True'
            s = son[mid][flag] == 'True'
            if g and s: a += 1
            elif g and not s: b += 1
            elif not g and s: c += 1
            else: d += 1

        mat = np.array([[d, c], [b, a]])
        cm_colors = [['#D6E4F0', '#9BB8D3'], ['#9BB8D3', '#2B5797']]

        t = ax.table(
            cellText=[[f'{mat[0,0]:,}', f'{mat[0,1]:,}'],
                      [f'{mat[1,0]:,}', f'{mat[1,1]:,}']],
            rowLabels=['Gem: 0', 'Gem: 1'],
            colLabels=['Son: 0', 'Son: 1'],
            cellColours=cm_colors, loc='center', cellLoc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        t.scale(1, 1.8)
        for (row, col), cell in t.get_celld().items():
            if row == 0 or col == -1:
                cell.set_text_props(fontweight='bold', fontsize=9)
                cell.set_facecolor('#E8E8E8')
            else:
                val = mat[row - 1, col]
                cell.set_text_props(fontweight='bold',
                                    color='white' if val > 3000 else 'black')
            cell.set_edgecolor('#BBBBBB')
        ax.set_title(name, fontweight='bold', fontsize=11, pad=8)
        ax.axis('off')

    fig.tight_layout(pad=1.5)
    fig.savefig(out, facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# FIGURE 8 — Temporal Dynamics (Series A)
# ═══════════════════════════════════════════════════════════════

def fig8_temporal(agr, out):
    a_rows = [r for r in agr if r['series'] == 'A']
    rounds = sorted(set(int(r['round']) for r in a_rows))

    colors = {
        'is_phatic': '#5B9BD5', 'is_meta': '#ED7D31', 'is_lead': '#70AD47',
        'is_arch': '#C00000', 'is_agree': '#7030A0', 'is_comp': '#A5A5A5'
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for flag in TRAIT_FLAGS + ['is_comp']:
        rates = []
        for rd in rounds:
            rd_rows = [r for r in a_rows if int(r['round']) == rd]
            n = len(rd_rows)
            rates.append(sum(1 for r in rd_rows if r[flag] == 'True') / n if n else 0)
        ls = '--' if flag == 'is_comp' else '-'
        ax.plot(rounds, rates, f'o{ls}', color=colors[flag],
                label=flag.replace('is_', '').capitalize(), markersize=4, lw=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Flag prevalence')
    ax.set_title('Temporal Dynamics — Series A (Baseline)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=3, loc='upper right')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# FIGURE 9 — Compensation Across Series
# ═══════════════════════════════════════════════════════════════

def fig9_comp_series(agr, out):
    series_order = ['A', 'E', 'H', 'J', 'I', 'G', 'C', 'K2', 'K3', 'K1']
    series_labels = [
        'A\nBaseline', 'E\nShuffle', 'H\nFestival', 'J\n20 rnd',
        'I\nHi temp', 'G\nEnglish', 'C\nReal nm',
        'K2\nNo ID', 'K3\nEmpty', 'K1\nNo peers'
    ]
    series_colors = {
        'A': '#5B9BD5', 'E': '#5B9BD5', 'H': '#FFC000', 'J': '#FFC000',
        'I': '#FFC000', 'G': '#FFC000', 'C': '#ED7D31',
        'K2': '#70AD47', 'K3': '#70AD47', 'K1': '#70AD47'
    }
    avgs = []
    for s in series_order:
        sr = [r for r in agr if r['series'] == s]
        comp = sum(1 for r in sr if r['is_comp'] == 'True')
        runs = len(set(r['file'] for r in sr))
        avgs.append(comp / runs if runs > 0 else 0)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(series_order)), avgs,
                  color=[series_colors[s] for s in series_order],
                  edgecolor='white', width=0.65)
    ax.set_xticks(range(len(series_order)))
    ax.set_xticklabels(series_labels, fontsize=7.5)
    ax.set_ylabel('Avg compensation events per run')
    ax.set_title('Compensatory Responses Across Conditions (Clean Data)',
                 fontsize=12, fontweight='bold')
    for b, v in zip(bars, avgs):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05,
                    f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# FIGURE 10 — Label Aggregation Sensitivity
# ═══════════════════════════════════════════════════════════════

def fig10_sensitivity(gem, son, out):
    n = len(gem)
    inter_pct, union_pct = [], []
    for f in SENS_FLAGS:
        inter = sum(1 for m in gem if gem[m][f] == 'True' and son[m][f] == 'True')
        union = sum(1 for m in gem if gem[m][f] == 'True' or son[m][f] == 'True')
        inter_pct.append(inter / n * 100)
        union_pct.append(union / n * 100)

    x = np.arange(len(SENS_FLAGS))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - 0.18, inter_pct, 0.32,
           label='Conservative ($\\cap$)', color='#5B9BD5')
    ax.bar(x + 0.18, union_pct, 0.32,
           label='Union ($\\cup$)', color='#ED7D31')
    ax.set_xticks(x)
    ax.set_xticklabels(SENS_LABELS)
    ax.set_ylabel('Prevalence (%)')
    ax.set_title('Label Aggregation Sensitivity',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--data-dir', default='./data',
                        help='Directory containing CSV files')
    parser.add_argument('--output-dir', default='./figures',
                        help='Directory to save figures')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {data_dir}/")
    agr = load_csv(os.path.join(data_dir, 'coded_v4_agreed.csv'))
    gem = load_csv_dict(os.path.join(data_dir, 'coded_v4_gemini.csv'))
    son = load_csv_dict(os.path.join(data_dir, 'coded_v4_sonnet.csv'))
    print(f"  agreed: {len(agr)} rows")
    print(f"  gemini: {len(gem)} rows")
    print(f"  sonnet: {len(son)} rows")
    print()

    # Generate all figures
    figures = [
        ("fig0_pipeline.png",    lambda o: fig0_pipeline(o)),
        ("fig1_profiles.png",    lambda o: fig1_profiles(agr, o)),
        ("fig2_isolation.png",   lambda o: fig2_isolation(agr, o)),
        ("fig3_compensation.png",lambda o: fig3_compensation(agr, o)),
        ("fig4_name_bias.png",   lambda o: fig4_name_bias(agr, o)),
        ("fig5_heterogeneity.png", lambda o: fig5_heterogeneity(agr, o)),
        ("fig6_ablation.png",    lambda o: fig6_ablation(agr, o)),
        ("fig7_confusion.png",   lambda o: fig7_confusion(gem, son, o)),
        ("fig8_temporal.png",    lambda o: fig8_temporal(agr, o)),
        ("fig9_comp_series.png", lambda o: fig9_comp_series(agr, o)),
        ("fig10_sensitivity.png", lambda o: fig10_sensitivity(gem, son, o)),
    ]

    for name, func in figures:
        path = os.path.join(out_dir, name)
        func(path)
        print(f"  ✓ {name}")

    print(f"\nAll {len(figures)} figures saved to {out_dir}/")


if __name__ == '__main__':
    main()
