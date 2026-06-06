from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    Helper for residual connection used after each sub-layer.
      - Project `output` through `dense_layer`, apply `dropout`, then add to `input`.
      - Layer norm is NOT applied here (GPT-2 uses pre-norm: the norm is applied
        *before* the sub-layer in forward(), not here).
    """
    return input + dropout(dense_layer(output))


  def forward(self, hidden_states, attention_mask, return_attn_probs=False):
    """
    GPT-2 transformer block (pre-LayerNorm variant).

    If return_attn_probs=True, additionally returns this layer's attention
    probabilities of shape [bs, num_heads, seq_len, seq_len].
    """
    # --- Self-attention sub-layer (pre-norm + residual) ---
    normed = self.attention_layer_norm(hidden_states)
    if return_attn_probs:
      attn_output, attn_probs = self.self_attention(normed, attention_mask, return_attn_probs=True)
    else:
      attn_output = self.self_attention(normed, attention_mask)
      attn_probs = None

    hidden_states = self.add(
        hidden_states, attn_output, self.attention_dense, self.attention_dropout
    )

    # --- Feed-forward sub-layer (pre-norm + residual) ---
    normed = self.out_layer_norm(hidden_states)
    ff_output = self.interm_af(self.interm_dense(normed))
    hidden_states = self.add(
        hidden_states, ff_output, self.out_dense, self.out_dropout
    )

    if return_attn_probs:
      return hidden_states, attn_probs
    return hidden_states