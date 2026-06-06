"""
extract_attention.py
====================
Attention 시각화를 위한 attention weight 추출 스크립트.

목적:
    Step 2 결과(step2-outputs.csv)에서 권위 prefix가 붙은 샘플을 골라:
      - Flip 발생 (authority prefix 때문에 예측이 틀어진 샘플)
      - Flip 미발생 (예측이 그대로인 샘플)
    두 그룹에 대해 마지막 토큰이 prefix 토큰들에 얼마나 attend 하는지 추출.

입력:
    - data/step2-inputs.csv      (sent_id, dataset, condition, prefix, input_text 등)
    - data/step2-outputs.csv     (sent_id, dataset, condition, true_label, pred_label, ...)
    - cfimdb-classifier.pt 또는 imdb-classifier.pt  (선택, fine-tuned 모델)

출력:
    - data/attention_results.csv (sent_id, dataset, condition, flip, attention_to_prefix, ...)
    - data/attention_arrays.npz  (per-sample raw attention arrays)
"""
import os
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer
from tqdm import tqdm

# Local project imports (must run from project root)
from models.gpt2 import GPT2Model


# ============================================================
# 설정
# ============================================================
SEED = 11711
LAST_LAYER_IDX = -1   # Notion: "마지막 토큰이 prefix 토큰들에 얼마나 attend"
                       # = 가장 마지막 layer 사용
DEVICE = 'cpu'         # Mac 환경. inference만이므로 OK


# ============================================================
# 핵심: classifier 정의 (classifier.py의 동일 구조)
# ============================================================
class GPT2SentimentClassifier(torch.nn.Module):
    """
    classifier.py의 GPT2SentimentClassifier와 동일 구조.
    return_attn_probs=True를 지원하도록 forward만 확장.
    """
    def __init__(self, num_labels, hidden_dropout_prob=0.1, hidden_size=768):
        super().__init__()
        self.num_labels = num_labels
        self.gpt = GPT2Model.from_pretrained()
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, return_attn_probs=False):
        gpt_output = self.gpt(input_ids, attention_mask, return_attn_probs=return_attn_probs)
        last_token = gpt_output['last_token']
        pooled = self.dropout(last_token)
        logits = self.classifier(pooled)
        if return_attn_probs:
            return logits, gpt_output['attn_probs']
        return logits


# ============================================================
# 데이터 로드 + Flip 샘플 식별
# ============================================================
def load_and_identify_flips(inputs_path, outputs_path, dataset_name):
    """
    Step 2 입력/출력 CSV를 로드하고 join하여, 권위 조건에서 flip 발생/미발생을 식별.

    Flip 정의:
      - authority_pos/neg 조건에서 pred_label != true_label
      - 단, 같은 sent_id의 control 조건에서는 pred_label == true_label (=정답)
        → "원래는 맞췄는데 권위 prefix 때문에 틀린" 케이스
    """
    inputs = pd.read_csv(inputs_path, sep='\t')
    outputs = pd.read_csv(outputs_path, sep='\t')

    # sent_id + condition + dataset 으로 join
    df = inputs.merge(
        outputs[['sent_id', 'dataset', 'condition', 'true_label', 'pred_label']],
        on=['sent_id', 'dataset', 'condition'],
        how='inner',
    )
    df = df[df['dataset'] == dataset_name].reset_index(drop=True)

    # control 조건에서 정답인 sent_id 만 후보로 (flip 정의 보장)
    control_correct = df[(df['condition'] == 'control')
                         & (df['true_label'] == df['pred_label'])]['sent_id'].unique()
    df = df[df['sent_id'].isin(control_correct)].reset_index(drop=True)

    # 권위 조건만 추출
    auth = df[df['condition'].isin(['authority_pos', 'authority_neg'])].copy()
    auth['flip'] = (auth['true_label'] != auth['pred_label']).astype(int)
    return auth


# ============================================================
# Attention 추출
# ============================================================
def extract_attention_for_sample(model, tokenizer, prefix_text, full_input_text,
                                  layer_idx=LAST_LAYER_IDX):
    """
    하나의 샘플에 대해 마지막 토큰이 prefix 토큰들에 얼마나 attend 하는지 계산.

    Returns:
      attn_to_prefix_total: 마지막 토큰 → prefix tokens 의 sum attention (스칼라)
      attn_to_prefix_mean:  마지막 토큰 → prefix tokens 의 mean attention (스칼라)
      attn_per_prefix_token: prefix 각 토큰별 attention (list)
      n_prefix_tokens, n_total_tokens
    """
    if not isinstance(prefix_text, str) or len(prefix_text) == 0:
        return None  # prefix 없음(control). 권위 조건만 처리하면 발생하지 않음

    # Prefix 단독 토큰 수
    prefix_ids = tokenizer(prefix_text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
    n_prefix = len(prefix_ids)

    # 전체 입력 토큰화
    enc = tokenizer(full_input_text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = enc['input_ids'].to(DEVICE)
    attn_mask = enc['attention_mask'].to(DEVICE)

    # Forward + attention 추출
    with torch.no_grad():
        _, attn_probs_list = model(input_ids, attn_mask, return_attn_probs=True)

    # 마지막 layer attention: [1, num_heads, seq_len, seq_len]
    attn = attn_probs_list[layer_idx][0]  # [num_heads, seq_len, seq_len]

    # 마지막 non-pad 위치
    last_pos = int(attn_mask.sum().item()) - 1
    # 마지막 토큰의 query → 모든 토큰 key, heads 간 평균
    last_token_attn = attn[:, last_pos, :].mean(dim=0)  # [seq_len]

    # 안전: n_prefix가 last_pos 보다 큰 경우 방지
    n_prefix_used = min(n_prefix, last_pos)
    attn_to_prefix_vec = last_token_attn[:n_prefix_used]

    return {
        'attn_to_prefix_total': attn_to_prefix_vec.sum().item(),
        'attn_to_prefix_mean':  attn_to_prefix_vec.mean().item(),
        'attn_per_prefix_token': attn_to_prefix_vec.cpu().numpy().tolist(),
        'n_prefix_tokens': n_prefix_used,
        'n_total_tokens': last_pos + 1,
    }


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['cfimdb', 'imdb'])
    parser.add_argument('--inputs',  default='data/step2-inputs.csv')
    parser.add_argument('--outputs', default='data/step2-outputs.csv')
    parser.add_argument('--ckpt', default=None,
                        help='fine-tuned model (.pt). None이면 base GPT-2 사용')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--out_csv',     default='data/attention_results.csv')
    parser.add_argument('--out_npz',     default='data/attention_arrays.npz')
    parser.add_argument('--max_samples', type=int, default=200,
                        help='Flip / 비Flip 그룹에서 각각 최대 추출 샘플 수')
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"=== Attention 시각화 추출 ({args.dataset}) ===")
    print(f"  inputs : {args.inputs}")
    print(f"  outputs: {args.outputs}")
    print(f"  ckpt   : {args.ckpt or 'base GPT-2'}")

    # 1. 모델 초기화
    classifier = GPT2SentimentClassifier(num_labels=args.num_labels).to(DEVICE).eval()
    if args.ckpt:
        if not os.path.exists(args.ckpt):
            print(f"⚠️  ckpt 파일이 없습니다: {args.ckpt}")
            print("    → base GPT-2 weights로 진행 (attention pattern은 거의 동일)")
        else:
            state = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
            if isinstance(state, dict) and 'model' in state:
                state = state['model']
            classifier.load_state_dict(state, strict=False)
            print(f"  ✅ fine-tuned weight 로드: {args.ckpt}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Flip 샘플 식별
    auth_df = load_and_identify_flips(args.inputs, args.outputs, args.dataset)
    print(f"\n[{args.dataset}] 권위 조건 샘플: {len(auth_df)}개")
    print(auth_df['flip'].value_counts().rename({0: '비Flip', 1: 'Flip'}).to_string())

    # 3. Flip / 비Flip 그룹에서 샘플링 (재현 가능)
    n_flip   = len(auth_df[auth_df['flip']==1])
    n_noflip = len(auth_df[auth_df['flip']==0])
    flip_df    = auth_df[auth_df['flip'] == 1].sample(
        min(n_flip, args.max_samples), random_state=SEED)
    noflip_df  = auth_df[auth_df['flip'] == 0].sample(
        min(n_noflip, args.max_samples), random_state=SEED)
    target_df  = pd.concat([flip_df, noflip_df]).reset_index(drop=True)
    print(f"\n추출 대상: Flip {len(flip_df)} / 비Flip {len(noflip_df)}")

    # 4. Attention 추출
    rows = []
    raw_attn = {}
    for _, r in tqdm(target_df.iterrows(), total=len(target_df), desc='extracting'):
        info = extract_attention_for_sample(
            classifier, tokenizer,
            prefix_text=str(r['prefix']) if pd.notna(r['prefix']) else '',
            full_input_text=str(r['input_text']),
        )
        if info is None:
            continue
        key = f"{r['sent_id']}::{r['condition']}"
        rows.append({
            'sent_id':              r['sent_id'],
            'dataset':              r['dataset'],
            'condition':            r['condition'],
            'true_label':           r['true_label'],
            'pred_label':           r['pred_label'],
            'flip':                 r['flip'],
            'attn_to_prefix_total': info['attn_to_prefix_total'],
            'attn_to_prefix_mean':  info['attn_to_prefix_mean'],
            'n_prefix_tokens':      info['n_prefix_tokens'],
            'n_total_tokens':       info['n_total_tokens'],
        })
        raw_attn[key] = np.array(info['attn_per_prefix_token'], dtype=np.float32)

    # 5. 저장
    os.makedirs('data', exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.out_csv, sep='\t', index=False)
    np.savez(args.out_npz, **raw_attn)

    # 6. 요약
    print(f"\n✅ 저장: {args.out_csv} ({len(result_df)}행)")
    print(f"✅ 저장: {args.out_npz} (raw attention vectors)")
    print("\n=== Flip vs 비Flip: 마지막 토큰이 prefix에 attend 한 정도 ===")
    summary = result_df.groupby('flip')['attn_to_prefix_total'].agg(['count', 'mean', 'std']).round(4)
    summary.index = summary.index.map({0: '비Flip', 1: 'Flip'})
    print(summary.to_string())


if __name__ == '__main__':
    main()