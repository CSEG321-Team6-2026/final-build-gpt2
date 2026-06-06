"""
visualize_attention.py
======================
extract_attention.py가 추출한 결과를 시각화하는 스크립트.

생성 그림:
  1. fig_cfimdb.png  — CFIMDB Flip vs 비Flip box/strip plot + 통계 검정
  2. fig_imdb.png    — IMDB    Flip vs 비Flip box/strip plot + 통계 검정
  3. fig_compare.png — 두 데이터셋 비교 막대 그래프 (mean ± SE)

입력:
  data/attention_results_cfimdb.csv
  data/attention_results_imdb.csv

사용법:
  python visualize_attention.py
"""
import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', font_scale=1.05)


def welch_t(flip_values, noflip_values):
    """Welch's t-test (분산 비동질 가정). 두 그룹 평균 비교."""
    return stats.ttest_ind(flip_values, noflip_values, equal_var=False)


def fmt_p(p):
    if p < 0.001: return 'p < 0.001'
    if p < 0.01:  return f'p = {p:.3f}'
    return f'p = {p:.3f}'


def plot_single_dataset(df, title, save_path):
    """단일 데이터셋: Flip vs 비Flip의 attn_to_prefix_total 분포 비교."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    plot_df = df.copy()
    plot_df['group'] = plot_df['flip'].map({0: 'No Flip', 1: 'Flip'})

    sns.boxplot(
        data=plot_df, x='group', y='attn_to_prefix_total',
        order=['No Flip', 'Flip'],
        width=0.5, showfliers=False, ax=ax,
        palette={'No Flip': '#a3c4f3', 'Flip': '#ff9a9a'},
    )
    sns.stripplot(
        data=plot_df, x='group', y='attn_to_prefix_total',
        order=['No Flip', 'Flip'],
        size=4, alpha=0.55, jitter=0.18, color='0.25', ax=ax,
    )

    means = plot_df.groupby('group')['attn_to_prefix_total'].mean()
    stds  = plot_df.groupby('group')['attn_to_prefix_total'].std()
    ns    = plot_df.groupby('group').size()
    ax.scatter(['No Flip', 'Flip'],
               [means['No Flip'], means['Flip']],
               marker='D', s=80, color='black', zorder=10, label='mean')

    f_vals = plot_df[plot_df['group'] == 'Flip']['attn_to_prefix_total'].values
    n_vals = plot_df[plot_df['group'] == 'No Flip']['attn_to_prefix_total'].values
    t_res = welch_t(f_vals, n_vals)
    diff_pct = (means['Flip'] - means['No Flip']) / means['No Flip'] * 100

    info = (f"No Flip: n={ns['No Flip']}, mean={means['No Flip']:.4f}, std={stds['No Flip']:.4f}\n"
            f"Flip   : n={ns['Flip']}, mean={means['Flip']:.4f}, std={stds['Flip']:.4f}\n"
            f"Δ = +{diff_pct:.1f}%   Welch's t = {t_res.statistic:.2f}, {fmt_p(t_res.pvalue)}")

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel("Last token's attention to prefix\n(sum over prefix tokens, last layer)", fontsize=11)
    ax.text(0.02, 0.98, info, transform=ax.transAxes,
            fontsize=9.5, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='0.7', alpha=0.95))
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  ✅ saved: {save_path}")
    return {'mean_flip': means['Flip'], 'mean_noflip': means['No Flip'],
            'std_flip': stds['Flip'], 'std_noflip': stds['No Flip'],
            'n_flip': int(ns['Flip']), 'n_noflip': int(ns['No Flip']),
            't': t_res.statistic, 'p': t_res.pvalue, 'diff_pct': diff_pct}


def plot_compare(stats_cfimdb, stats_imdb, save_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    datasets = ['CFIMDB', 'IMDB']
    means_noflip = [stats_cfimdb['mean_noflip'], stats_imdb['mean_noflip']]
    means_flip   = [stats_cfimdb['mean_flip'],   stats_imdb['mean_flip']]
    se_noflip = [stats_cfimdb['std_noflip']/np.sqrt(stats_cfimdb['n_noflip']),
                 stats_imdb['std_noflip']  /np.sqrt(stats_imdb['n_noflip'])]
    se_flip   = [stats_cfimdb['std_flip']/np.sqrt(stats_cfimdb['n_flip']),
                 stats_imdb['std_flip']  /np.sqrt(stats_imdb['n_flip'])]

    x = np.arange(len(datasets))
    w = 0.35
    b1 = ax.bar(x - w/2, means_noflip, w, yerr=se_noflip, capsize=5,
                label='No Flip', color='#a3c4f3', edgecolor='black')
    b2 = ax.bar(x + w/2, means_flip,   w, yerr=se_flip,   capsize=5,
                label='Flip',    color='#ff9a9a', edgecolor='black')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9.5)

    for i, s in enumerate([stats_cfimdb, stats_imdb]):
        ax.text(i, max(means_noflip[i], means_flip[i]) + 0.02,
                f"Δ +{s['diff_pct']:.1f}%\n{fmt_p(s['p'])}",
                ha='center', fontsize=10, fontweight='bold',
                color='#c0392b')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Last token's attention to prefix (mean ± SE)", fontsize=11)
    ax.set_title("Attention to prefix: Flip vs No Flip (Authority condition)",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(means_flip + means_noflip) * 1.45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  ✅ saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfimdb', default='data/attention_results_cfimdb.csv')
    p.add_argument('--imdb',   default='data/attention_results_imdb.csv')
    p.add_argument('--outdir', default='figures')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=== Attention 시각화 ===")
    cf = pd.read_csv(args.cfimdb, sep='\t')
    im = pd.read_csv(args.imdb,   sep='\t')
    print(f"  CFIMDB: {len(cf)}행")
    print(f"  IMDB  : {len(im)}행")

    print("\n[1/3] CFIMDB ...")
    s_cf = plot_single_dataset(cf, 'CFIMDB: Attention to Prefix',
                                os.path.join(args.outdir, 'fig_cfimdb.png'))

    print("\n[2/3] IMDB ...")
    s_im = plot_single_dataset(im, 'IMDB: Attention to Prefix',
                                os.path.join(args.outdir, 'fig_imdb.png'))

    print("\n[3/3] Comparison ...")
    plot_compare(s_cf, s_im, os.path.join(args.outdir, 'fig_compare.png'))

    print("\n=== 통계 요약 ===")
    print(f"{'':10}{'mean(NoFlip)':>14}{'mean(Flip)':>14}{'Δ%':>10}{'t':>8}{'p':>12}")
    for name, s in [('CFIMDB', s_cf), ('IMDB', s_im)]:
        print(f"{name:10}{s['mean_noflip']:>14.4f}{s['mean_flip']:>14.4f}"
              f"{s['diff_pct']:>9.1f}%{s['t']:>8.2f}   {fmt_p(s['p'])}")


if __name__ == '__main__':
    main()