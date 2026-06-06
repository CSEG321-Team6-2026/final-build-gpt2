import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# 와이드 포맷 데이터 로드
pivot_df = pd.read_csv("data/step3_processed_wide.csv")

# 데이터셋별(CFIMDB, IMDB)로 분리하여 검증 수행
for dataset in pivot_df['dataset'].unique():
    subset = pivot_df[pivot_df['dataset'] == dataset]
    print(f"\n==================== {dataset.upper()} 데이터셋 McNemar's Test ====================")
    
    # 5가지 모든 조건 검증
    all_conditions = ['general_pos', 'general_neg', 'authority_pos', 'authority_neg', 'neutral']
    
    for cond in all_conditions:
        if f'pred_label_{cond}' not in subset.columns:
            print(f"[경고] {cond} 조건에 해당하는 컬럼을 찾을 수 없습니다. 건너뜁니다.")
            continue
            
        # Control 예측값과 각 조건 예측값 간의 2x2 교차표 생성
        contingency_table = pd.crosstab(subset['pred_label_control'], subset[f'pred_label_{cond}'])
        contingency_table = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        
        # McNemar's Test 실행 
        result = mcnemar(contingency_table, exact=False, correction=True)
        
        print(f"\n[조건: {cond} vs control]")
        print(f"  - 검정 통계량 (Statistic): {result.statistic:.4f}")
        print(f"  - p-value: {result.pvalue:.4e}")
        
        # 유의수준 0.05 기준 판정
        if result.pvalue < 0.05:
            print("  ➔ 결과: p-value < 0.05 이므로 통계적으로 매우 유의미한 예측 변화입니다.")
        else:
            print("  ➔ 결과: p-value >= 0.05 이므로 통계적으로 유의미한 변화가 아닙니다.")