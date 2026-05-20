from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PWLayer(nn.Module):
    """Parametric whitening layer used by the UniSRec adaptor."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.linear = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE adaptor from UniSRec."""

    def __init__(self, n_exps: int, input_size: int, output_size: int, dropout: float = 0.0, noise: bool = True):
        super().__init__()
        self.n_exps = n_exps
        self.noisy_gating = noise
        self.experts = nn.ModuleList([PWLayer(input_size, output_size, dropout) for _ in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x: torch.Tensor) -> torch.Tensor:
        logits = x @ self.w_gate
        if self.noisy_gating and self.training:
            noise_std = F.softplus(x @ self.w_noise) + 1e-2
            logits = logits + torch.randn_like(logits) * noise_std
        return F.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.noisy_top_k_gating(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return (gates.unsqueeze(-1) * expert_outputs).sum(dim=-2)


class _BaseSemanticModel(nn.Module):
    """Shared semantic projection, scoring, and embedding replacement utilities."""

    def __init__(self, hidden_units: int, pretrained_item_embeddings: torch.Tensor):
        super().__init__()
        if pretrained_item_embeddings is None:
            raise ValueError("pretrained_item_embeddings is required.")
        self.hidden_units = hidden_units
        self.pretrained_dim = pretrained_item_embeddings.shape[1]
        self.pretrained_item_embedding = self._build_embedding(pretrained_item_embeddings)

    @staticmethod
    def _build_embedding(pretrained_item_embeddings: torch.Tensor) -> nn.Embedding:
        weights = pretrained_item_embeddings.clone().float()
        weights[0] = 0.0
        return nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)

    def load_new_pretrain_embeddings(self, pretrained_item_embeddings: torch.Tensor) -> None:
        device = next(self.parameters()).device
        self.pretrained_item_embedding = self._build_embedding(pretrained_item_embeddings.to(device))

    def score_items(self, user_rep: torch.Tensor) -> torch.Tensor:
        item_rep = self.projection_layer(self.pretrained_item_embedding.weight)
        logits = user_rep @ item_rep.T
        logits[:, 0] = -1e9
        return logits

    def forward(self, item_seq: torch.Tensor, is_target_domain: bool = False, is_sem_baseline: bool = False) -> torch.Tensor:
        user_rep = self.encode_sequence(item_seq, is_sem_baseline=is_sem_baseline)
        return self.score_items(user_rep)

    def predict(
        self,
        item_seq: torch.Tensor,
        candidate_items: torch.Tensor | None = None,
        is_target_domain: bool = False,
        is_sem_baseline: bool = False,
    ) -> torch.Tensor:
        logits = self.forward(item_seq, is_target_domain=is_target_domain, is_sem_baseline=is_sem_baseline)
        if candidate_items is None:
            return logits
        return torch.gather(logits, dim=1, index=candidate_items)

    def project_items_for_alignment(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.domain_alignment_projection_layer(self.pretrained_item_embedding(item_ids))

    def project_raw_for_alignment(self, raw_embeddings: torch.Tensor) -> torch.Tensor:
        return self.domain_alignment_projection_layer(raw_embeddings)

    def get_raw_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.pretrained_item_embedding(item_ids)


class _TransformerBackbone(_BaseSemanticModel):
    def __init__(
        self,
        hidden_units: int,
        max_seq_length: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        pretrained_item_embeddings: torch.Tensor,
        causal: bool,
        projection: str = "linear",
        n_exps: int = 8,
    ):
        super().__init__(hidden_units, pretrained_item_embeddings)
        self.max_seq_length = max_seq_length
        self.causal = causal

        if projection == "moe":
            self.projection_layer = MoEAdaptorLayer(n_exps, self.pretrained_dim, hidden_units, dropout_rate)
            self.domain_alignment_projection_layer = MoEAdaptorLayer(n_exps, self.pretrained_dim, hidden_units, dropout_rate)
        else:
            self.projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
            self.domain_alignment_projection_layer = nn.Linear(self.pretrained_dim, hidden_units)

        self.merge_layer = nn.Linear(hidden_units * 2, hidden_units)
        self.pos_embedding = nn.Embedding(max_seq_length, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_units,
                nhead=num_heads,
                dim_feedforward=hidden_units * 4,
                dropout=dropout_rate,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-6)

    def encode_sequence(self, item_seq: torch.Tensor, is_sem_baseline: bool = False) -> torch.Tensor:
        device = item_seq.device
        batch_size, seq_len = item_seq.shape

        raw_emb = self.pretrained_item_embedding(item_seq)
        valid_mask = (item_seq != 0).float().unsqueeze(-1)

        rec_emb = self.projection_layer(raw_emb) * valid_mask
        if is_sem_baseline:
            seq_emb = rec_emb
        else:
            align_emb = self.domain_alignment_projection_layer(raw_emb) * valid_mask
            seq_emb = self.merge_layer(torch.cat([rec_emb, align_emb], dim=-1)) * valid_mask

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        out = self.dropout(seq_emb + self.pos_embedding(positions))

        attention_mask = None
        if self.causal:
            attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        padding_mask = item_seq == 0
        for layer in self.layers:
            out = layer(out, src_mask=attention_mask, src_key_padding_mask=padding_mask)

        out = self.layer_norm(out) * valid_mask
        return out[:, -1, :]


class SASRecWithDomainAlignment(_TransformerBackbone):
    def __init__(
        self,
        hidden_units: int,
        max_seq_length: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        pretrained_item_embeddings: torch.Tensor,
        **kwargs,
    ):
        super().__init__(
            hidden_units=hidden_units,
            max_seq_length=max_seq_length,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            pretrained_item_embeddings=pretrained_item_embeddings,
            causal=True,
        )


class BERT4RecWithDomainAlignment(_TransformerBackbone):
    def __init__(
        self,
        hidden_units: int,
        max_seq_length: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        pretrained_item_embeddings: torch.Tensor,
        **kwargs,
    ):
        super().__init__(
            hidden_units=hidden_units,
            max_seq_length=max_seq_length,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            pretrained_item_embeddings=pretrained_item_embeddings,
            causal=False,
        )


class UniSRecWithDomainAlignment(_TransformerBackbone):
    def __init__(
        self,
        hidden_units: int,
        max_seq_length: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        pretrained_item_embeddings: torch.Tensor,
        n_exps: int = 8,
        **kwargs,
    ):
        super().__init__(
            hidden_units=hidden_units,
            max_seq_length=max_seq_length,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            pretrained_item_embeddings=pretrained_item_embeddings,
            causal=True,
            projection="moe",
            n_exps=n_exps,
        )


class GRU4RecWithDomainAlignment(_BaseSemanticModel):
    def __init__(
        self,
        hidden_units: int,
        max_seq_length: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        pretrained_item_embeddings: torch.Tensor,
        **kwargs,
    ):
        super().__init__(hidden_units, pretrained_item_embeddings)
        self.max_seq_length = max_seq_length
        self.projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
        self.domain_alignment_projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
        self.merge_layer = nn.Linear(hidden_units * 2, hidden_units)

        self.input_norm = nn.LayerNorm(hidden_units, eps=1e-6)
        self.output_norm = nn.LayerNorm(hidden_units, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(
            input_size=hidden_units,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

    def encode_sequence(self, item_seq: torch.Tensor, is_sem_baseline: bool = False) -> torch.Tensor:
        raw_emb = self.pretrained_item_embedding(item_seq)
        valid_mask = (item_seq != 0).float().unsqueeze(-1)

        rec_emb = self.projection_layer(raw_emb)
        if is_sem_baseline:
            seq_emb = rec_emb
        else:
            align_emb = self.domain_alignment_projection_layer(raw_emb)
            seq_emb = self.merge_layer(torch.cat([rec_emb, align_emb], dim=-1))

        seq_emb = self.input_norm(seq_emb)
        seq_emb = F.gelu(seq_emb)
        seq_emb = self.dropout(seq_emb * valid_mask)

        output, _ = self.gru(seq_emb)
        seq_lengths = (item_seq != 0).sum(dim=1)
        last_index = (seq_lengths - 1).clamp(min=0)
        user_rep = output[torch.arange(item_seq.size(0), device=item_seq.device), last_index]
        return self.output_norm(user_rep)


MODEL_REGISTRY = {
    "sasrec": SASRecWithDomainAlignment,
    "bert4rec": BERT4RecWithDomainAlignment,
    "gru4rec": GRU4RecWithDomainAlignment,
    "unisrec": UniSRecWithDomainAlignment,
}


def build_model(
    model_name: str,
    hidden_units: int,
    max_seq_length: int,
    num_heads: int,
    num_layers: int,
    dropout_rate: float,
    pretrained_item_embeddings: torch.Tensor,
    n_exps: int = 8,
):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](
        hidden_units=hidden_units,
        max_seq_length=max_seq_length,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        pretrained_item_embeddings=pretrained_item_embeddings,
        n_exps=n_exps,
    )
