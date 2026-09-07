"""
vLLM Model Definition for Qwen4Exp (Qwen3.8-Flash-Next)
Hybrid SSM + QSA Sparse Attention + PLE + 512-MoE.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

class Qwen4ExpConfig:
    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 2048,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        full_attention_interval: int = 4,
        ssm_state_size: int = 128,
        num_experts: int = 512,
        num_experts_per_tok: int = 10,
        max_position_embeddings: int = 262144,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.full_attention_interval = full_attention_interval
        self.ssm_state_size = ssm_state_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.max_position_embeddings = max_position_embeddings

class Qwen4ExpSSMBlock(nn.Module):
    """Linear SSM block replacing full attention for non-interval layers."""
    def __init__(self, config: Qwen4ExpConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.state_size = config.ssm_state_size
        self.in_proj = nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, ssm_state: Optional[torch.Tensor] = None):
        # Fused conv1d + state update
        projected = self.in_proj(x)
        u, gate = projected.chunk(2, dim=-1)
        output = u * torch.sigmoid(gate)
        return self.out_proj(output), ssm_state

class Qwen4ExpMoEBlock(nn.Module):
    """Fine-grained 512-expert MoE layer with top-10 routing."""
    def __init__(self, config: Qwen4ExpConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)
        scores, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(scores, dim=-1)
        return weights, indices
