## implementation_test/attention_test.py
## attention 구현을 테스트하기 위한 테스트 코드
## python -m imp_test.attention_test

import torch
from modules.attention import CausalSelfAttention


bs, num_heads, seq_len, head_size = 2, 8, 10, 64
config = type('Config', (), {
    'num_attention_heads': num_heads,
    'hidden_size': num_heads * head_size,
    'attention_probs_dropout_prob': 0.1
})


model = CausalSelfAttention(config)
model.eval()

# dummy 생성
q = torch.randn(bs, num_heads, seq_len, head_size)
k = torch.randn(bs, num_heads, seq_len, head_size)
v = torch.randn(bs, num_heads, seq_len, head_size)
mask = torch.zeros(bs, 1, 1, seq_len) 

output = model.attention(k, q, v, mask)

# shape 확인
print(f"출력 크기: {output.shape}") # 결과 [2, 10, 512] (bs, seq_len, hidden_size)

# masking 확인
q2, k2 = q.clone(), k.clone()
q2[:, :, -1, :] += 10.0 
k2[:, :, -1, :] += 10.0

out1 = model.attention(k, q, v, mask) # 원래 데이터 결과
out2 = model.attention(k2, q2, v, mask) # 마지막 단어만 바꾼 데이터 결과

diff = torch.abs(out1[:, :-1, :] - out2[:, :-1, :]).max().item()
print(f"미래 토큰 변경 시 과거 데이터 영향도: {diff}")

if diff < 1e-5:
    print("✅ 테스트 통과")
else:
    print("❌ 테스트 실패")