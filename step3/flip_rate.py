import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('data/step2-outputs.csv', sep='\t')

# Wide Format으로 변환 
pivot_df = df.pivot(
    index=['sent_id', 'dataset', 'true_label'], 
    columns='condition', 
    values=['pred_label', 'prob_positive', 'prob_negative']
).reset_index()

# 컬럼 이름 flattening 
pivot_df.columns = [
    f"{col[0]}_{col[1]}" if col[1] else col[0] 
    for col in pivot_df.columns
]

print("변환된 데이터 컬럼 목록:", pivot_df.columns.tolist())

# Flip 여부 정의 
prefix_conditions = ['general_pos', 'general_neg', 'authority_pos', 'authority_neg', 'neutral']

for cond in prefix_conditions:
    if f'pred_label_{cond}' in pivot_df.columns:
        pivot_df[f'flip_{cond}'] = (pivot_df['pred_label_control'] != pivot_df[f'pred_label_{cond}']).astype(int)

# 데이터셋별 / 조건별 Flip Rate 계산
results = []
available_conditions = [c for c in prefix_conditions if f'flip_{c}' in pivot_df.columns]

for dataset in pivot_df['dataset'].unique():
    subset = pivot_df[pivot_df['dataset'] == dataset]
    
    for cond in available_conditions:
        flip_rate = subset[f'flip_{cond}'].mean() * 100
        results.append({
            'Dataset': dataset.upper(),
            'Condition': cond,
            'Flip Rate (%)': flip_rate
        })

summary_df = pd.DataFrame(results)
print("\n=== 최종 Flip Rate 집계 결과 ===")
print(summary_df)

# 시각화 
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# 데이터셋별로 묶어서 조건별 Flip Rate 그리기
ax = sns.barplot(
    x='Dataset', 
    y='Flip Rate (%)', 
    hue='Condition', 
    data=summary_df, 
    palette='Set2'
)

for p in ax.patches:
    height = p.get_height()
    if height > 0: # 0%인 경우 제외
        ax.annotate(f"{height:.1f}%", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', 
                    xytext=(0, 8), 
                    textcoords='offset points',
                    fontsize=10, fontweight='bold')

plt.title("Flip Rate Comparison by Prefix Conditions", fontsize=14, pad=15)
plt.ylabel("Flip Rate (%)", fontsize=12)
plt.xlabel("Dataset", fontsize=12)
plt.ylim(0, summary_df['Flip Rate (%)'].max() + 10)
plt.legend(title="Prefix Condition", bbox_to_anchor=(1.05, 1), loc='upper left')

# 그래프 저장 및 출력
plt.savefig("data/step3_flip_rate_chart.png", dpi=300, bbox_inches='tight')
plt.show()

# Ablation을 위해 데이터프레임을 CSV로 임시 저장
pivot_df.to_csv("data/step3_processed_wide.csv", index=False)
print("\n wide format 파일이 'data/step3_processed_wide.csv'에 저장되었습니다.")