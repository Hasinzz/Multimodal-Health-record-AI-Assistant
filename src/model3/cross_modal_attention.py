from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AttentionFusionResult:
    fused_embedding: torch.Tensor
    attention_weights: Optional[torch.Tensor]
    status: str


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int = 4):
        super().__init__()
        safe_heads = num_heads if embedding_dim % num_heads == 0 and embedding_dim >= num_heads else 1
        self.embedding_dim = embedding_dim
        self.project_image = nn.Linear(embedding_dim, embedding_dim)
        self.project_text = nn.Linear(embedding_dim, embedding_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=safe_heads, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, image_embedding: torch.Tensor, text_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_token = self.project_image(image_embedding).unsqueeze(1)
        text_token = self.project_text(text_embedding).unsqueeze(1)
        sequence = torch.cat([image_token, text_token], dim=1)
        attended, attention_weights = self.attention(sequence, sequence, sequence)
        fused = attended.mean(dim=1)
        fused = self.output(fused)
        return fused, attention_weights


def fuse_embeddings(
    image_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    model: Optional[CrossModalAttentionFusion] = None,
) -> AttentionFusionResult:
    if image_embedding.dim() == 1:
        image_embedding = image_embedding.unsqueeze(0)
    if text_embedding.dim() == 1:
        text_embedding = text_embedding.unsqueeze(0)

    embedding_dim = min(image_embedding.shape[-1], text_embedding.shape[-1])
    image_embedding = image_embedding[..., :embedding_dim]
    text_embedding = text_embedding[..., :embedding_dim]

    if model is None:
        model = CrossModalAttentionFusion(embedding_dim=embedding_dim)

    model.eval()
    with torch.no_grad():
        fused, attention_weights = model(image_embedding, text_embedding)

    return AttentionFusionResult(
        fused_embedding=fused.squeeze(0),
        attention_weights=attention_weights,
        status="experimental_untrained_attention",
    )
