import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 데이터 로드
df = pd.read_csv('data/step2-outputs.csv', sep='\t')

# ── 1. true_label 기준으로 분리 ──────────────────────────────────────────────

pos_df = df[df['true_label'] == 1].copy()
neg_df = df[df['true_label'] == 0].copy()

# ── 2. 각 그룹별 pivot ───────────────────────────────────────────────────────

pos_wide = pos_df[pos_df['condition'].isin(['neutral', 'general_neg', 'authority_neg'])].pivot(
    index=['sent_id', 'dataset', 'true_label'],
    columns='condition',
    values='prob_positive'
).reset_index()
pos_wide.columns.name = None

neg_wide = neg_df[neg_df['condition'].isin(['neutral', 'general_pos', 'authority_pos'])].pivot(
    index=['sent_id', 'dataset', 'true_label'],
    columns='condition',
    values='prob_negative'
).reset_index()
neg_wide.columns.name = None

# ── 3. Δp 계산 ───────────────────────────────────────────────────────────────

pos_wide['delta_p_general']   = pos_wide['general_neg']   - pos_wide['neutral']
pos_wide['delta_p_authority'] = pos_wide['authority_neg'] - pos_wide['neutral']

neg_wide['delta_p_general']   = neg_wide['general_pos']   - neg_wide['neutral']
neg_wide['delta_p_authority'] = neg_wide['authority_pos'] - neg_wide['neutral']

print("=== Positive 그룹 Δp 요약 ===")
print(pos_wide.groupby('dataset')[['delta_p_general', 'delta_p_authority']].describe().round(4))
print("\n=== Negative 그룹 Δp 요약 ===")
print(neg_wide.groupby('dataset')[['delta_p_general', 'delta_p_authority']].describe().round(4))

# ── 4. 바이올린 플롯 시각화 ──────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Δp Distribution by Condition and Dataset", fontsize=14)

datasets = ['cfimdb', 'imdb']
groups = [
    (pos_wide, 'Positive (true_label=1)', 'general_neg', 'authority_neg'),
    (neg_wide, 'Negative (true_label=0)', 'general_pos', 'authority_pos'),
]

for col_idx, dataset in enumerate(datasets):
    for row_idx, (wide_df, title, general_label, authority_label) in enumerate(groups):
        ax = axes[row_idx][col_idx]
        subset = wide_df[wide_df['dataset'] == dataset]

        plot_data = pd.DataFrame({
            'Δp': list(subset['delta_p_general'].values) + list(subset['delta_p_authority'].values),
            'Condition': [general_label] * len(subset) + [authority_label] * len(subset)
        })

        sns.violinplot(data=plot_data, x='Condition', y='Δp', ax=ax,
                       hue='Condition', palette='Set2', inner=None, legend=False)
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8)

        # 중앙값(흰 점)과 평균(검은 다이아몬드) 표시
        for i, (col, label) in enumerate([('delta_p_general', general_label),
                                           ('delta_p_authority', authority_label)]):
            vals = subset[col].values
            ax.scatter(i, np.median(vals), color='white', zorder=3, s=40)
            ax.scatter(i, np.mean(vals),   color='black', zorder=3, s=40, marker='D')

        # 범례 (한 번만)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=8, label='Median'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
                   markersize=8, label='Mean'),
        ]
        ax.legend(handles=legend_elements, fontsize=8)
        ax.set_title(f"{dataset.upper()} — {title}")
        ax.set_ylabel("Δp")
        ax.set_xlabel("Condition")

plt.tight_layout()
plt.savefig("data/step3_delta_p_violin.png", dpi=300, bbox_inches='tight')
plt.show()
print("\n[완료] 바이올린 플롯이 'data/step3_delta_p_violin.png'에 저장되었습니다.")
