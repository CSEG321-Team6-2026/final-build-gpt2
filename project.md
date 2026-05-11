# final-build-gpt2

CSE5321 기초자연어처리 (Intro to NLP) — Final Project  
Sogang University, Spring 2026 | Prof. Hwaran Lee

---

## 프로젝트 개요

Stanford CS224N의 Default Final Project를 기반으로 GPT-2를 직접 구현하고, 이를 활용해 downstream task를 탐구합니다.

- **Step 1**: GPT-2 아키텍처 구현 + Sentiment Analysis fine-tuning
- **Step 2**: 자체 Research Question 설정 및 실험

스타터 코드: https://github.com/cfifty/public_cs224n_gpt  
과제 문서: [CS224n Default Final Project: Build GPT-2](https://web.stanford.edu/class/cs224n/project_w25/CS_224n__Default_Final_Project__Build_GPT_2.pdf)

---

## Step 1 구현 목록

| 파일 | 구현 내용 | 난이도 | 담당 |
|------|----------|--------|------|
| `modules/attention.py` | Masked Multi-Head Self-Attention | 높음 | 팀원 1 |
| `modules/gpt2_layer.py` | GPT2Layer.add(), forward() | 중간 | 팀원 2 |
| `models/gpt2.py` | embed() | 중간 | 팀원 2 |
| `optimizer.py` | AdamW.step() | 중간 | 팀원 3 |
| `classifier.py` | GPT2SentimentClassifier | 중간 | 팀원 3 |


**[Step 1 실행 환경 안내]**

각자 담당 파일 구현 및 디버깅은 로컬에서 진행합니다.
단, 아래 단계는 팀원 구현 완료 여부에 따라 순서가 있습니다.

**1. 개인 디버깅** (로컬)
각자 담당 파일을 로컬에서 구현 및 디버깅합니다.

**2. `optimizer_test.py`** (로컬, 팀원 3 완료 후)
Adam Optimizer 구현 검증. 팀원 3 혼자 실행 가능합니다.

**3. `sanity_check.py`** (로컬, 팀원 1 + 2 완료 후)
GPT-2 구현 검증. 팀원 1과 2의 구현이 모두 완료되어야 실행 가능합니다.

**4. `classifier.py` 학습** (Colab, 팀원 1 + 2 + 3 완료 후)
전체 팀원 구현이 합쳐져야 실행 가능합니다. GPU가 필요하므로 Colab에서 진행합니다.

즉, Step 1 최종 검증은 팀원 모두의 구현이 완료되어야 가능합니다.
각자 진행 상황을 팀원들과 공유해주세요.

---

## Step 2 — Research Question 설정 및 실험

Step 1 구현을 마친 후, 팀이 자체적으로 Research Question을 설정하고 실험을 진행합니다.

### 목표

> "What is your hypothesis? And how will you set up the experiment setting to show your hypothesis?"

결과가 좋지 않아도 괜찮습니다. 왜 안 됐는지 분석하는 것이 중요합니다.

### 방향 예시

**성능 개선 관점**

| 방법 | 핵심 논문 |
|------|----------|
| LoRA (Parameter-efficient fine-tuning) | Hu et al., 2021 |
| DPO (Direct Preference Optimization) | Rafailov et al., 2023 |
| SMART Regularization | Jiang et al., 2021 |
| FlashAttention / Sliding Window Attention | Dao et al., 2022 |
| Quantization | GPTQ, Quadapter |
| Second-order Optimizer (Shampoo, SOAP) | Martens & Grosse, 2015 |
| Hyperparameter Search / Ensembling | — |

**태스크 확장 관점**

- 한국어 downstream task (예: NSMC 감성분석 데이터셋 활용)
- Paraphrase Detection / Sonnet Generation 외 새로운 태스크

### 실험 대상 태스크

Step 1에서 구현한 세 가지 태스크를 기반으로 실험합니다.

| 태스크 | 데이터셋 | 평가 지표 |
|--------|---------|----------|
| Sentiment Analysis | SST, CFIMDB | Accuracy |
| Paraphrase Detection | Quora (train 141,506 / dev 20,215 / test 40,431) | Accuracy |
| Sonnet Generation | Shakespeare Sonnets (train 143 / test 12) | chrF score |

### Leaderboard 제출 (Paraphrase / Sonnet)

- Dev: 제출 횟수 무제한
- **Test: 최대 3회** — 신중하게 사용할 것

### Research Question 확정 시 이 섹션을 업데이트할 것

```
- 가설:
- 실험 설계:
- 결과 요약:
- 분석:
```

---

## 시작하기

팀 리포지토리(`final-build-gpt2`)에는 이미 스타터 코드가 push되어 있습니다.  
스타터 코드(과제 깃허브) 클론 여부에 따라 아래 중 해당하는 방법을 따라주세요.

### Case 1 — 스타터 코드를 아직 클론하지 않은 경우

팀 리포지토리를 바로 클론합니다.

```bash
git clone https://github.com/YOUR_ORG/final-build-gpt2.git
cd final-build-gpt2
```

### Case 2 — 스타터 코드를 이미 클론한 경우

기존에 클론한 디렉토리의 remote를 팀 리포지토리로 교체합니다.

```bash
cd public_cs224n_gpt
git remote remove origin
git remote add origin https://github.com/YOUR_ORG/final-build-gpt2.git
git pull origin main --allow-unrelated-histories
```

---

### 환경 세팅

```bash
source setup.sh          # conda 환경 cs224n_dfp 생성
conda activate cs224n_dfp
```

### 구현 검증

```bash
python3 sanity_check.py      # GPT-2 구현 검증
python3 optimizer_test.py    # Adam Optimizer 검증
```

### Sentiment Analysis 학습

```bash
# Last-linear-layer 모드
python3 classifier.py --fine-tune-mode last-linear-layer --batch_size 8 --lr 1e-5 --epochs 5

# Full fine-tuning 모드
python3 classifier.py --fine-tune-mode full-model --batch_size 8 --lr 1e-5 --epochs 5
```

---

## 개발 환경

- **로컬 머신**: 코드 작성 및 디버깅 (GPU 없이)
- **Colab / GCP**: 디버깅 완료 후 실제 학습

> 로컬에서 충분히 디버깅한 뒤 Colab/GCP로 옮길 것. GPU 시간 낭비 방지.

---

## 협업 규칙

### 브랜치 전략

```
main         ← 완성된 코드만 merge
dev          ← 통합 브랜치
feat/이름-기능  ← 개인 작업 브랜치
```

작업 흐름:
1. `dev`에서 개인 브랜치 생성: `git checkout -b feat/이름-기능`
2. 작업 후 `dev`에 Pull Request
3. 팀원 1명 이상 리뷰 후 merge
4. `dev` 안정화되면 `main`에 merge

### 커밋 메시지 규칙

```
feat: 새 기능 추가
fix: 버그 수정
refactor: 코드 개선
docs: 문서 수정
exp: 실험 코드 (Step 2)
```

예시:
```
feat: implement causal self-attention
fix: fix positional embedding indexing bug
exp: add LoRA fine-tuning for paraphrase detection
```

### 코드 리뷰

- PR은 merge 전 최소 1명 리뷰
- 리뷰어는 48시간 이내 응답

---

## 디렉토리 구조

> ✏️ 표시된 파일은 코드 구현이 필요한 파일입니다.

```
final-build-gpt2/
├── models/
│   ├── base_gpt.py
│   └── gpt2.py                ✏️ embed() 구현 (Token + Positional Embedding)
├── modules/
│   ├── attention.py           ✏️ CausalSelfAttention.attention() 구현 (Masked Multi-Head Attention)
│   └── gpt2_layer.py          ✏️ GPT2Layer.add(), forward() 구현
├── data/                      # 데이터셋
│   ├── ids-sst-{train,dev,test-student}.csv
│   ├── ids-cfimdb-{train,dev,test-student}.csv
│   ├── quora-{train,dev,test-student}.csv
│   ├── sonnets.txt
│   ├── sonnets_held_out.txt
│   ├── sonnets_held_out_dev.txt
│   └── TRUE_sonnets_held_out_dev.txt
├── predictions/               # 모델 예측 결과 (자동 생성)
│   └── README
├── classifier.py              ✏️ GPT2SentimentClassifier 구현
├── paraphrase_detection.py
├── sonnet_generation.py
├── optimizer.py               ✏️ AdamW.step() 구현
├── optimizer_test.py          # Optimizer 검증 스크립트
├── optimizer_test.npy         # Optimizer 검증용 데이터
├── sanity_check.py            # GPT-2 구현 검증 스크립트
├── datasets.py                # 데이터셋 로딩 유틸리티
├── evaluation.py              # 평가 유틸리티
├── config.py                  # 모델 설정
├── utils.py                   # 유틸리티 함수
├── prepare_submit.py          # 제출 파일 생성 스크립트
├── setup.sh                   # 환경 세팅 스크립트
├── env.yml                    # conda 환경 파일
├── LICENSE
├── README.md                  # 스타터 코드 원본 README
└── PROJECT.md                 # 팀 프로젝트 문서 (이 파일)
```

---

## 주요 일정

| 기간 | 할 일 | 구분 |
|------|-------|------|
| 지금 ~ 5/25 | Step 1 구현 (attention, gpt2_layer, embed, optimizer, classifier 구현 및 sanity check 통과) | Step 1 |
| 5/25 ~ 6/1 | Step 1 마무리 + Step 2 방향 확정 (Sentiment Analysis 학습 완료, dev accuracy 기준치 달성, Research Question 확정) | 전환 |
| 6/1 ~ 6/8 | Step 2 실험 (Paraphrase Detection / Sonnet Generation 실험, dev leaderboard 제출) | Step 2 |
| 6/8 ~ 6/10 | 발표 준비 + test leaderboard 제출 (Test 제출 최대 3회 — 신중하게 사용) | 마감 |
| **6/10 오후 3시** | **발표 파일 제출 — Cyber Campus** | 마감 |
| 6/10 or 15 | Final Presentation (10분 발표) | 발표 |
| 6/10 ~ 6/24 | Report 작성 (6페이지 이내, 한국어/영어 모두 가능) | Step 2 |
| **6/24 23:59** | **Report + 코드 최종 제출 — Cyber Campus** | 마감 |

---

## LLM 사용 기록

이 수업은 LLM 사용을 허용하나, 사용 내역을 반드시 문서화해야 합니다.  
각자 사용한 LLM 도구, 질문 내용, 활용 방식, 수정 사항을 기록해주세요.
