"""
build_input_csv.py
==================
2단계 권위 편향(authority bias) 실험 입력 CSV 생성 스크립트.

목적:
    dev set 샘플(CFIMDB/IMDB)의 각 본문에 대해, 본문 감성과 반대 방향의
    prefix를 4가지 조건(control / neutral / general / authority)으로 붙여
    실험 입력 데이터를 생성한다.

조건 설계:
    - negative 본문(gold_label=0) → positive 방향 prefix를 붙임
        control / neutral / general_pos / authority_pos
    - positive 본문(gold_label=1) → negative 방향 prefix를 붙임
        control / neutral / general_neg / authority_neg

    각 (본문 × 조건)마다 해당 pool에서 prefix 1개를 무작위 선택(seed=11711).
    control은 prefix 없음(원문 그대로).

입력:
    - data/cfimdb-dev-sampled.csv  (TSV: id, sentence, sentiment)
    - data/imdb-dev-sampled.csv    (TSV: id, sentence, sentiment)
    - prefix_pool.py               (authority_pos/neg, general_pos/neg, neutral)

출력:
    - data/step2-inputs.csv  (TSV)
    컬럼: sent_id, dataset, gold_label, condition, prefix, input_text

    데이터 수: 2 dataset × 200 sample × 4 condition = 1,600행

재현성:
    random.seed(11711)
"""

import os
import random
import pandas as pd

from prefix_pool import (
    authority_pos, authority_neg,
    general_pos, general_neg,
    neutral,
)

# ============================================================
# 설정
# ============================================================
SEED = 11711

DATA_DIR = "data"
OUTPUT_PATH = os.path.join(DATA_DIR, "step2-inputs.csv")

SOURCES = [
    {"name": "cfimdb", "path": os.path.join(DATA_DIR, "cfimdb-dev-sampled.csv")},
    {"name": "imdb",   "path": os.path.join(DATA_DIR, "imdb-dev-sampled.csv")},
]

# 본문 감성(gold_label)에 따라 붙일 prefix 방향 정의.
# negative 본문(0)에는 positive 방향, positive 본문(1)에는 negative 방향.
# 각 조건은 (조건명, pool) 형태. control은 pool이 None(=prefix 없음).
PREFIX_PLAN = {
    0: [  # negative 본문 → positive 방향 prefix
        ("control",       None),
        ("neutral",       neutral),
        ("general_pos",   general_pos),
        ("authority_pos", authority_pos),
    ],
    1: [  # positive 본문 → negative 방향 prefix
        ("control",       None),
        ("neutral",       neutral),
        ("general_neg",   general_neg),
        ("authority_neg", authority_neg),
    ],
}


# ============================================================
# 핵심 함수
# ============================================================
def build_rows(rng: random.Random) -> list:
    """모든 데이터셋·본문·조건 조합에 대한 입력 행 생성."""
    rows = []

    for src in SOURCES:
        name = src["name"]
        path = src["path"]

        if not os.path.exists(path):
            print(f"  ⚠️  파일 없음 — 건너뜀: {path}")
            continue

        df = pd.read_csv(path, sep="\t")
        df["sentiment"] = df["sentiment"].astype(int)
        print(f"[{name}] {len(df)}개 본문 로드")

        for _, row in df.iterrows():
            sent_id = row["id"]
            sentence = str(row["sentence"]).strip()
            gold = int(row["sentiment"])

            for condition, pool in PREFIX_PLAN[gold]:
                if pool is None:
                    # control: prefix 없음
                    prefix = ""
                    input_text = sentence
                else:
                    # pool에서 무작위 1개 선택
                    prefix = rng.choice(pool)
                    input_text = f"{prefix} {sentence}"

                rows.append({
                    "sent_id":    sent_id,
                    "dataset":    name,
                    "gold_label": gold,
                    "condition":  condition,
                    "prefix":     prefix,
                    "input_text": input_text,
                })

    return rows


# ============================================================
# 실행
# ============================================================
def main():
    print("=" * 60)
    print(f"Step 2 Input CSV 생성 (seed = {SEED})")
    print("=" * 60)

    # 재현 가능한 독립 난수 생성기
    rng = random.Random(SEED)

    rows = build_rows(rng)

    out_df = pd.DataFrame(
        rows,
        columns=["sent_id", "dataset", "gold_label",
                 "condition", "prefix", "input_text"],
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, sep="\t", index=False)

    # 결과 요약
    print("\n" + "=" * 60)
    print(f"✅ 저장: {OUTPUT_PATH}")
    print(f"총 {len(out_df)}행 생성")
    print("\n[조건별 행 수]")
    print(out_df.groupby(["dataset", "condition"]).size().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()