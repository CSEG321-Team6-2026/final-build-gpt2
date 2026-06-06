import pandas as pd

# 와이드 포맷 데이터 로드
pivot_df = pd.read_csv("data/step3_processed_wide.csv")

print("==================== 역방향 Flip 분석 ====================\n")

# 분석 결과를 담을 리스트
anomaly_summary = []

# 데이터셋별로 분석
for dataset in pivot_df['dataset'].unique():
    subset = pivot_df[pivot_df['dataset'] == dataset]
    print(f"▶ 데이터셋: {dataset.upper()}")
    
    # ----------------------------------------------------
    # 실제 정답이 부정이고, 원문도 부정으로 잘 맞췄는데,
    # 부정 자극을 주니까 갑자기 긍정으로 오판한 경우
    # ----------------------------------------------------
    # 일반 부정 자극
    gen_neg_anomaly = subset[
        (subset['true_label'] == 0) & 
        (subset['pred_label_control'] == 0) & 
        (subset['pred_label_general_neg'] == 1)
    ]
    # 권위 부정 자극
    auth_neg_anomaly = subset[
        (subset['true_label'] == 0) & 
        (subset['pred_label_control'] == 0) & 
        (subset['pred_label_authority_neg'] == 1)
    ]
    
    # ----------------------------------------------------
    # 실제 정답이 긍정이고, 원문도 긍정으로 잘 맞췄는데,
    # 긍정 자극을 주니까 갑자기 부정으로 오판한 경우
    # ----------------------------------------------------
    # 일반 긍정 자극
    gen_pos_anomaly = subset[
        (subset['true_label'] == 1) & 
        (subset['pred_label_control'] == 1) & 
        (subset['pred_label_general_pos'] == 0)
    ]
    # 권위 긍정 자극
    auth_pos_anomaly = subset[
        (subset['true_label'] == 1) & 
        (subset['pred_label_control'] == 1) & 
        (subset['pred_label_authority_pos'] == 0)
    ]
    
    # 출력 및 결과 저장
    print(f"  [부정 원문 ➔ 부정 자극 주니 '긍정'으로 뒤집힘]")
    print(f"    - general_neg 조건 에러 샘플 수: {len(gen_neg_anomaly)}개")
    print(f"    - authority_neg 조건 에러 샘플 수: {len(auth_neg_anomaly)}개")
    
    print(f"  [긍정 원문 ➔ 긍정 자극 주니 '부정'으로 뒤집힘]")
    print(f"    - general_pos 조건 에러 샘플 수: {len(gen_pos_anomaly)}개")
    print(f"    - authority_pos 조건 에러 샘플 수: {len(auth_pos_anomaly)}개")
    print("-" * 60)
    