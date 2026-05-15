## imp_test/classifier_test.py
## GPT2SentimentClassifier 구현을 테스트하기 위한 테스트 코드
## (GPT2Model을 mock으로 대체하여 classifier.py만 단독 검증)
## python -m imp_test.classifier_test

import torch
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

# --- 설정 ---
NUM_LABELS = 5       # SST: 5개 클래스
HIDDEN_SIZE = 768    # GPT2 hidden size 고정
BS = 2               # 배치 사이즈
SEQ_LEN = 16         # 임의 시퀀스 길이

config = SimpleNamespace(
    num_labels=NUM_LABELS,
    hidden_size=HIDDEN_SIZE,
    hidden_dropout_prob=0.1,
    fine_tune_mode='last-linear-layer',
)


def make_mock_gpt():
    """GPT2Model.from_pretrained()을 대체하는 mock 객체 생성"""
    mock_gpt = MagicMock()
    # forward() 호출 시 last_token: [BS, HIDDEN_SIZE] 반환
    mock_gpt.return_value = {'last_token': torch.randn(BS, HIDDEN_SIZE)}
    # parameters()는 학습 가능한 파라미터 2개를 가진 이터레이터 반환
    mock_gpt.parameters = MagicMock(return_value=iter([
        torch.nn.Parameter(torch.randn(1)),
        torch.nn.Parameter(torch.randn(1)),
    ]))
    return mock_gpt


# -------------------------------------------------------
# 테스트 1: 출력 shape 확인
# forward() 결과가 [bs, num_labels] = [2, 5] 인지 확인
# -------------------------------------------------------
with patch('classifier.GPT2Model') as MockGPT2Model:
    mock_gpt = make_mock_gpt()
    MockGPT2Model.from_pretrained.return_value = mock_gpt

    from classifier import GPT2SentimentClassifier
    model = GPT2SentimentClassifier(config)
    model.eval()

    input_ids = torch.randint(0, 50257, (BS, SEQ_LEN))
    attn_mask = torch.ones(BS, SEQ_LEN, dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, attn_mask)

    assert logits.shape == (BS, NUM_LABELS), \
        f"❌ shape 오류: 예상 {(BS, NUM_LABELS)}, 실제 {logits.shape}"
    print(f"출력 크기: {logits.shape}")
    print("✅ 테스트 1 통과: 출력 shape 정확")


# -------------------------------------------------------
# 테스트 2: last-linear-layer 모드에서 GPT 파라미터 동결 확인
# -------------------------------------------------------
with patch('classifier.GPT2Model') as MockGPT2Model:
    mock_gpt = make_mock_gpt()
    MockGPT2Model.from_pretrained.return_value = mock_gpt

    model = GPT2SentimentClassifier(config)

    for param in model.gpt.parameters():
        assert not param.requires_grad, "❌ GPT 파라미터가 동결되지 않았습니다"

    cls_grads = [p.requires_grad for p in model.classifier.parameters()]
    assert all(cls_grads), "❌ 분류기 파라미터가 학습 가능하지 않습니다"
    print("✅ 테스트 2 통과: last-linear-layer 파라미터 동결 정확")


# -------------------------------------------------------
# 테스트 3: full-model 모드에서 GPT 파라미터 학습 가능 확인
# -------------------------------------------------------
config_full = SimpleNamespace(
    num_labels=NUM_LABELS,
    hidden_size=HIDDEN_SIZE,
    hidden_dropout_prob=0.1,
    fine_tune_mode='full-model',
)

with patch('classifier.GPT2Model') as MockGPT2Model:
    mock_gpt = make_mock_gpt()
    MockGPT2Model.from_pretrained.return_value = mock_gpt

    model_full = GPT2SentimentClassifier(config_full)

    for param in model_full.gpt.parameters():
        assert param.requires_grad, "❌ full-model 모드에서 GPT 파라미터가 학습 불가 상태입니다"
    print("✅ 테스트 3 통과: full-model 파라미터 학습 가능 확인")


# -------------------------------------------------------
# 테스트 4: eval 모드 — dropout 비활성화 확인
# eval 모드: 동일 입력 → 동일 출력 (결정적)
# -------------------------------------------------------
with patch('classifier.GPT2Model') as MockGPT2Model:
    mock_gpt = make_mock_gpt()
    fixed_last_token = torch.randn(BS, HIDDEN_SIZE)
    mock_gpt.return_value = {'last_token': fixed_last_token}
    MockGPT2Model.from_pretrained.return_value = mock_gpt

    model = GPT2SentimentClassifier(config)
    model.eval()

    with torch.no_grad():
        out1 = model(input_ids, attn_mask)
        out2 = model(input_ids, attn_mask)

    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-6, f"❌ eval 모드에서 출력이 달라짐 (diff={diff})"
    print(f"eval 모드 출력 차이: {diff:.2e}")
    print("✅ 테스트 4 통과: eval 모드 결정적 출력 확인")
