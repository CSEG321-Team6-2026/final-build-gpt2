"""
2단계 B 추론 스크립트

입력:  data/step2-inputs.csv  (TSV)
출력:  data/step2-outputs.csv (TSV)
"""

import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

from classifier import GPT2SentimentClassifier

# ============================================================
# 설정
# ============================================================
INPUT_PATH  = "data/step2-inputs.csv"
OUTPUT_PATH = "data/step2-outputs.csv"

MODEL_PATHS = {
    "cfimdb": "cfimdb-classifier.pt",
    "imdb":   "imdb-classifier.pt",
}

BATCH_SIZE = 8


# ============================================================
# Dataset
# ============================================================
class InferenceDataset(Dataset):
    """step2-inputs.csv 한 dataset(cfimdb/imdb)의 행을 받아 토크나이징."""

    def __init__(self, rows: list):
        self.rows = rows
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def collate_fn(self, batch):
        # input_text를 소문자로 통일 (classifier.py의 load_data와 동일하게)
        texts       = [row["input_text"].lower().strip() for row in batch]
        sent_ids    = [row["sent_id"]                    for row in batch]
        conditions  = [row["condition"]                  for row in batch]
        gold_labels = [int(row["gold_label"])            for row in batch]

        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {
            "token_ids":      torch.LongTensor(encoding["input_ids"]),
            "attention_mask": torch.LongTensor(encoding["attention_mask"]),
            "sent_ids":       sent_ids,
            "conditions":     conditions,
            "gold_labels":    gold_labels,
        }


# ============================================================
# 모델 로드
# ============================================================
def load_model(filepath: str, device: torch.device) -> GPT2SentimentClassifier:
    saved  = torch.load(filepath, map_location=device, weights_only=False)
    config = saved["model_config"]
    model  = GPT2SentimentClassifier(config)
    model.load_state_dict(saved["model"])
    model  = model.to(device)
    model.eval()
    print(f"  모델 로드 완료: {filepath}")
    return model


# ============================================================
# 추론
# ============================================================
def run_inference(model, dataloader, device) -> list:
    """
    반환: list of dict
        sent_id / condition / gold_label / pred_label / prob_positive / prob_negative
    """
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="추론 중"):
            b_ids  = batch["token_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)

            logits = model(b_ids, b_mask)           # (batch, num_labels)
            probs  = F.softmax(logits, dim=-1)      # (batch, num_labels)
            preds  = torch.argmax(probs, dim=-1)    # (batch,)

            probs = probs.cpu().numpy()
            preds = preds.cpu().numpy()

            for i in range(len(batch["sent_ids"])):
                # num_labels == 2: index 0 = negative, index 1 = positive
                results.append({
                    "sent_id":       batch["sent_ids"][i],
                    "condition":     batch["conditions"][i],
                    "gold_label":    batch["gold_labels"][i],
                    "pred_label":    int(preds[i]),
                    "prob_negative": float(probs[i][0]),
                    "prob_positive": float(probs[i][1]),
                })
    return results


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true",
                        help="GPU 사용 (Colab 실행 시 지정)")
    parser.add_argument("--debug", action="store_true",
                        help="각 dataset에서 앞 20행만 처리 (로컬 디버깅용)")
    args = parser.parse_args()

    device = torch.device("cuda") if (args.use_gpu and torch.cuda.is_available()) \
             else torch.device("cpu")
    print(f"device: {device}")

    # ── 입력 파일 로드 ──────────────────────────────────────
    df = pd.read_csv(INPUT_PATH, sep="\t")
    print(f"입력 파일 로드: {len(df)}행")

    if args.debug:
        df = df.groupby("dataset", group_keys=False).apply(lambda x: x.head(20))
        print(f"  [debug 모드] {len(df)}행으로 축소")

    all_rows = []

    # ── dataset별 처리 ──────────────────────────────────────
    for dataset_name, model_path in MODEL_PATHS.items():
        print(f"\n{'='*50}")
        print(f"[{dataset_name}] 처리 시작")

        subset = df[df["dataset"] == dataset_name].to_dict("records")
        if len(subset) == 0:
            print(f"  ⚠️  {dataset_name} 데이터 없음 — 건너뜀")
            continue

        model = load_model(model_path, device)

        dataset_obj = InferenceDataset(subset)
        dataloader  = DataLoader(
            dataset_obj,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=dataset_obj.collate_fn,
        )

        results = run_inference(model, dataloader, device)

        for r in results:
            r["dataset"] = dataset_name

        all_rows.extend(results)
        print(f"  [{dataset_name}] 완료: {len(results)}행")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 출력 파일 저장 ──────────────────────────────────────
    out_df = pd.DataFrame(all_rows)
    out_df = out_df.rename(columns={"gold_label": "true_label"})
    out_df = out_df[["sent_id", "dataset", "condition",
                     "true_label", "pred_label",
                     "prob_positive", "prob_negative"]]

    out_df.to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"\n{'='*50}")
    print(f"✅ 저장 완료: {OUTPUT_PATH}")
    print(f"총 {len(out_df)}행")

    # ── Sanity Check ────────────────────────────────────────
    print("\n[Sanity Check]")
    print(f"  총 행 수: {len(out_df)}")

    prob_sum = (out_df["prob_positive"] + out_df["prob_negative"])
    assert (prob_sum - 1.0).abs().max() < 1e-5, "❌ prob 합산 오류"
    print("  prob_positive + prob_negative == 1.0 ✅")

    print("\n[control 조건 샘플 10개]")
    print(out_df[out_df["condition"] == "control"][
        ["sent_id", "dataset", "true_label", "pred_label", "prob_positive"]
    ].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
