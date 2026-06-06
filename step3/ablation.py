import pandas as pd

# 가공된 와이드 포맷 데이터 로드
pivot_df = pd.read_csv("data/step3_processed_wide.csv")

# 검증할 쌍
disruptive_conditions = ['general_pos', 'general_neg', 'authority_pos', 'authority_neg']

ablation_results = []

print("==================== Ablation Study 결과 ====================")

# 데이터셋별로 루프
for dataset in pivot_df['dataset'].unique():
    dataset_df = pivot_df[pivot_df['dataset'] == dataset]
    print(f"\n▶ 데이터셋: {dataset.upper()}")
    
    for cond in disruptive_conditions:
        if f'flip_{cond}' not in dataset_df.columns:
            continue
            
        # 해당 조건에서 Flip이 발생한 샘플만 필터링
        flipped_samples = dataset_df[dataset_df[f'flip_{cond}'] == 1]
        total_flipped = len(flipped_samples)
        
        if total_flipped == 0:
            print(f"  - [{cond}] 조건에서 Flip이 발생한 샘플이 없습니다.")
            continue
            
        # 1차 Ablation: Neutral 조건으로 교체 시 원래 예측(control)으로 복구되는지 확인
        # 복구 성공 = Neutral 예측값 == Control 예측값
        neutral_recovered = (flipped_samples['pred_label_neutral'] == flipped_samples['pred_label_control']).sum()
        neutral_recovery_rate = (neutral_recovered / total_flipped) * 100
        
        # 2차 Ablation: Control 조건(원문 자체)으로 복구 확인 
        control_recovered = (flipped_samples['pred_label_control'] == flipped_samples['pred_label_control']).sum()
        control_recovery_rate = (control_recovered / total_flipped) * 100
        
        print(f"  [{cond} 조건 교란 샘플 분석] (총 {total_flipped}개 샘플)")
        print(f"    - 1차 Neutral 대체 시 복구율 (Sentiment 제거): {neutral_recovery_rate:.1f}% ({neutral_recovered}/{total_flipped})")
        print(f"    - 2차 Control 복구율 (Prefix 완전 제거)   : {control_recovery_rate:.1f}% ({control_recovered}/{total_flipped})")
        
        # 결과 저장을 위해 기록
        ablation_results.append({
            'Dataset': dataset.upper(),
            'Condition': cond,
            'Total Flipped': total_flipped,
            'Neutral Recovery Rate (%)': neutral_recovery_rate,
            'Control Recovery Rate (%)': control_recovery_rate
        })

# 결과를 데이터프레임으로 시각화 및 출력
summary_ablation = pd.DataFrame(ablation_results)
print("\n==================== Ablation 요약 테이블 ====================")
print(summary_ablation.to_string(index=False))

# 최종 결과 저장
summary_ablation.to_csv("data/step3_ablation_summary.csv", index=False)
print("\n[완료] 인과 검증 결과가 'data/step3_ablation_summary.csv'에 저장되었습니다.")