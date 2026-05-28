"""
dev_set_sampling.py
===================
Step 2 실험 파이프라인용 dev set 샘플 추출 스크립트.

목적:
    CFIMDB 및 IMDB dev set에서 각 레이블별 100개씩 무작위 추출.
    (negative 100 + positive 100 = 데이터셋당 200개, 총 400개)

입력:
    - data/ids-cfimdb-dev.csv  (TSV 형식: id, sentence, sentiment)
    - data/imdb-dev.csv        (TSV 형식: id, sentence, sentiment)

출력:
    - data/cfimdb-dev-sampled.csv  (TSV, 200행)
    - data/imdb-dev-sampled.csv    (TSV, 200행)

재현성:
    random.seed(11711) — 프로젝트 통일 seed
"""

import os
import random
import pandas as pd

# ============================================================
# 설정
# ============================================================
SEED = 11711
SAMPLES_PER_LABEL = 100  # 레이블별 추출 개수

DATA_DIR = "data"

SOURCES = [
    {
        "name": "cfimdb",
        "input":  os.path.join(DATA_DIR, "ids-cfimdb-dev.csv"),
        "output": os.path.join(DATA_DIR, "cfimdb-dev-sampled.csv"),
    },
    {
        "name": "imdb",
        "input":  os.path.join(DATA_DIR, "imdb-dev.csv"),
        "output": os.path.join(DATA_DIR, "imdb-dev-sampled.csv"),
    },
]


# ============================================================
# 핵심 함수
# ============================================================
def sample_dev_set(input_path: str, output_path: str, name: str) -> None:
    """레이블별 100개씩 무작위 추출하여 저장."""
    print(f"\n[{name.upper()}] 처리 시작")
    print(f"  입력: {input_path}")

    if not os.path.exists(input_path):
        print(f"  ⚠️  파일이 없습니다 — 건너뜀: {input_path}")
        return

    df = pd.read_csv(input_path, sep="\t")
    print(f"  전체 데이터: {len(df)}개")

    df["sentiment"] = df["sentiment"].astype(int)

    df_neg = df[df["sentiment"] == 0]
    df_pos = df[df["sentiment"] == 1]
    print(f"  negative: {len(df_neg)}개  /  positive: {len(df_pos)}개")

    if len(df_neg) < SAMPLES_PER_LABEL or len(df_pos) < SAMPLES_PER_LABEL:
        print(f"  ⚠️  레이블당 {SAMPLES_PER_LABEL}개 미만! 추출 불가.")
        return

    sampled_neg = df_neg.sample(n=SAMPLES_PER_LABEL, random_state=SEED)
    sampled_pos = df_pos.sample(n=SAMPLES_PER_LABEL, random_state=SEED)

    sampled = pd.concat([sampled_neg, sampled_pos], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)

    sampled.to_csv(output_path, sep="\t", index=False)

    print(f"  ✅ 저장: {output_path}")
    print(f"  추출 결과: 총 {len(sampled)}개 "
          f"(negative {(sampled['sentiment']==0).sum()} + "
          f"positive {(sampled['sentiment']==1).sum()})")


# ============================================================
# 실행
# ============================================================
def main():
    print("=" * 60)
    print("Dev Set Sampling (seed = {})".format(SEED))
    print("=" * 60)

    random.seed(SEED)

    for src in SOURCES:
        sample_dev_set(
            input_path=src["input"],
            output_path=src["output"],
            name=src["name"],
        )

    print("\n" + "=" * 60)
    print("✅ 전체 처리 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()