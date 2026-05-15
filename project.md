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

## 협업 규칙

### 브랜치 전략

```
main          ← 완성된 코드만 merge
dev           ← 통합 브랜치
{issue_number}/{description}  ← 개인 작업 브랜치 (ex. 1/add_attention )
```

작업 흐름:
1. 작업 전 GitHub에 Issue 작성
2. 개인 작업별 브랜치 생성: `git checkout -b {issue_number}/{description}`
3. 작업 후 `dev`에 Pull Request
4. PR 후 단톡에 공유
5. 팀원 1명 이상 리뷰 후 merge
6. `dev` 안정화되면 `main`에 merge

### 코드 리뷰
- PR은 merge 전 최소 1명 리뷰

---

## Step 1 구현 목록

| 파일 | 구현 내용 | 담당 | 이름 |
|------|----------|------|------|
| `modules/attention.py` | `attention()` | 팀원 1 | 박서연 |
| `modules/gpt2_layer.py` | `add()`, `forward()` | 팀원 2 | XIONG MINGXING |
| `models/gpt2.py` | `embed()`, `hidden_state_to_token()` | 팀원 2 | XIONG MINGXING |
| `optimizer.py` | `AdamW.step()` | 팀원 3 | 이서진 |
| `classifier.py` | `GPT2SentimentClassifier.__init__()`, `forward()` | 팀원 3 | 이서진 |


### [Step 1 실행 환경 안내]

각자 담당 파일 구현 및 디버깅은 로컬에서 진행합니다.

모든 팀원은 서로의 완료 여부와 무관하게 구현을 바로 시작할 수 있습니다.
단, 실행/학습은 아래 순서를 따릅니다.

**실행 순서**

1. **개인 디버깅** (로컬)
각자 담당 파일을 로컬에서 구현 및 디버깅합니다.

2. **`optimizer_test.py`** (로컬, 팀원 3 완료 후)
Adam Optimizer 구현 검증. 팀원 3 혼자 실행 가능합니다.

3. **`sanity_check.py`** (로컬, 팀원 1 + 2 완료 후)
GPT-2 구현 검증. 팀원 1과 2의 구현이 모두 완료되어야 실행 가능합니다. 팀원 3은 이 단계에서 자신의 구현이 전체 코드와 잘 연결되는지 확인할 수 있습니다.

4. **`classifier.py` 학습** (Colab, 팀원 1 + 2 + 3 완료 후)
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

## 제출 체크리스트

### 📦 Step 1 제출 — Gradescope `Default Final Project [Base]`

`classifier.py`를 두 모드로 모두 실행한 뒤, `prepare_submit.py`로 zip을 생성해 업로드합니다.

```bash
# 1. last-linear-layer 모드 학습 (Colab, GPU)
python3 classifier.py --fine-tune-mode last-linear-layer \
    --batch_size 64 --lr 1e-3 --epochs 10 --use_gpu

# 2. full-model 모드 학습 (Colab, GPU)
python3 classifier.py --fine-tune-mode full-model \
    --batch_size 8 --lr 1e-5 --epochs 10 --use_gpu

# 3. 제출 파일 생성
python3 prepare_submit.py
# → cs224n_default_final_project_submission.zip 생성됨
```

제출 전 아래 파일이 모두 존재하는지 확인할 것:

```
modules/attention.py
modules/gpt2_layer.py
models/base_gpt.py
models/gpt2.py
classifier.py
optimizer.py
predictions/last-linear-layer-sst-dev-out.csv
predictions/last-linear-layer-sst-test-out.csv
predictions/full-model-sst-dev-out.csv
predictions/full-model-sst-test-out.csv
predictions/last-linear-layer-cfimdb-dev-out.csv
predictions/last-linear-layer-cfimdb-test-out.csv
predictions/full-model-cfimdb-dev-out.csv
predictions/full-model-cfimdb-test-out.csv
```

**목표 성능 (dev 기준)**

| 모드 | SST | CFIMDB |
|------|-----|--------|
| last-linear-layer | ≥ 0.462 | ≥ 0.861 |
| full-model | ≥ 0.513 | ≥ 0.976 |


### 🏆 Step 2 Leaderboard 제출 — Gradescope

`prepare_submit.py` 실행 후 동일한 zip 파일을 각 leaderboard에 업로드합니다.

| 태스크 | Dev | Test | 제출 횟수 제한 |
|--------|-----|------|--------------|
| Paraphrase Detection | Default Final Project [Paraphrase Dev] | Default Final Project [Paraphrase Test] | Test **최대 3회** |
| Sonnet Generation | Default Final Project [Sonnet Dev] | Default Final Project [Sonnet Test] | Test **최대 3회** |

> ⚠️ Test 제출은 신중하게. 최종 성적은 가장 좋은 제출 기준 (가장 최근 X).


### 📝 최종 제출 — Cyber Campus (`6/24 23:59`)

| 항목 | 내용 |
|------|------|
| 코드 | `cs224n_default_final_project_submission.zip` |
| 보고서 | PDF, **참고문헌 제외 최대 6페이지**, 한국어/영어 모두 가능 |

**보고서에 포함할 내용**

- Step 1: GPT-2 구현 설명, Sentiment Analysis 실험 결과 (last-linear-layer vs full-model 비교)
- Step 2: Research Question, 가설, 실험 설계, 결과, 분석 (결과가 나빠도 왜 안 됐는지 분석이 중요)


### 🗓️ 발표 — Cyber Campus (`6/10 오후 3시` 파일 제출)

- 발표 시간: 10분
- 발표일: 6/10 또는 6/15 (미정)
- PPT 파일을 Cyber Campus에 제출

---

## 개발 환경

- **로컬 머신**: 코드 작성 및 디버깅 (GPU 없이)
- **Colab / GCP**: 디버깅 완료 후 실제 학습

> 로컬에서 충분히 디버깅한 뒤 Colab/GCP로 옮길 것. GPU 시간 낭비 방지.

---

## 주요 일정

| 날짜 | 할 일 | 담당 |
|------|-------|------|
| ~ 5/13 | `attention()` 구현 완료 | 팀원 1 |
| ~ 5/16 | `add()`, `forward()`, `embed()`, `hidden_state_to_token()` 구현 완료 | 팀원 2 |
| ~ 5/17 | `AdamW.step()`, `GPT2SentimentClassifier.__init__()`, `forward()` 구현 완료 및 `classifier.py` 학습 | 팀원 3 |
| 5/21 ~ | Step 2 시작 | 전체 |
| 5/26 | Step 2 구현/실험 시작 | 전체 |
| 5/29 ~ 6/1 | Step 2 실험 계속 (팀원 1 일정 바쁨) | 팀원 2, 3 |
| ~ 6/6 | Step 2 마무리 | 전체 |
| 6/7 ~ 6/8 | 리포트 작성 → 발표 준비 (PPT) | 전체 |
| **6/10 오후 3시** | **발표 파일 제출 — Cyber Campus** | 전체 |
| 6/10 or 15 | Final Presentation (10분 발표) | 전체 |
| **6/24 23:59** | **Report + 코드 최종 제출 — Cyber Campus** | 전체 |

