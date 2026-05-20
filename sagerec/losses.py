from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=1, eps=eps)
    y = F.normalize(y, p=2, dim=1, eps=eps)
    return x @ y.T


def unified_alignment_loss(
    mode: str,
    projected_embeddings: torch.Tensor,
    raw_embeddings: torch.Tensor,
    domain_labels: torch.Tensor,
    num_domains: int,
    source_domain_id: int,
    lambda_g: float = 0.001,
    gamma_g: float = 1.0,
    beta_id: float = 0.01,
    tau: float = 0.1,
    sim_threshold: float = 0.0,
):
    if mode not in {"sem", "recg", "sage"}:
        raise ValueError("mode must be one of: sem, recg, sage.")

    device = projected_embeddings.device
    zero = torch.tensor(0.0, device=device)

    if mode == "sem":
        return zero.requires_grad_(True), zero, zero, torch.tensor(float(lambda_g), device=device)

    if mode == "recg":
        domain_centers = []
        diversity = torch.tensor(0.0, device=device)

        for domain_id in range(num_domains):
            mask = domain_labels == domain_id
            if mask.any():
                domain_z = projected_embeddings[mask]
                domain_centers.append(domain_z.mean(dim=0, keepdim=True))
                probs = F.softmax(cosine_similarity(domain_z, domain_z) / tau, dim=1)
                diversity = diversity - (probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
            else:
                domain_centers.append(torch.zeros((1, projected_embeddings.size(1)), device=device))

        diversity = diversity / max(num_domains, 1)
        centers = torch.cat(domain_centers, dim=0)
        sim = cosine_similarity(projected_embeddings, centers)

        same_domain = torch.zeros_like(sim, dtype=torch.bool)
        for row, domain_id in enumerate(domain_labels):
            same_domain[row, int(domain_id.item())] = True

        sim = sim.masked_fill(same_domain, -float("inf"))
        inter_probs = F.softmax(sim / tau, dim=1)
        inter_entropy = -(inter_probs * torch.log(inter_probs + 1e-10)).sum(dim=1).mean()

        beta = lambda_g * (projected_embeddings.size(0) / (max(num_domains, 1) ** 3))
        gen_loss = -lambda_g * diversity + beta * inter_entropy
        return gen_loss, inter_entropy.detach(), diversity.detach(), torch.tensor(float(lambda_g), device=device)

    source_mask = domain_labels == source_domain_id
    target_mask = ~source_mask
    z_s, z_t = projected_embeddings[source_mask], projected_embeddings[target_mask]
    raw_s, raw_t = raw_embeddings[source_mask], raw_embeddings[target_mask]

    diversity = torch.tensor(0.0, device=device)
    for domain_z in (z_s, z_t):
        if domain_z.size(0) > 1:
            probs = F.softmax(cosine_similarity(domain_z, domain_z) / tau, dim=1)
            diversity = diversity - (probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
    diversity = diversity / 2.0

    sic = torch.tensor(0.0, device=device)
    if z_s.size(0) > 0 and z_t.size(0) > 0:
        z_s = F.normalize(z_s, p=2, dim=1)
        z_t = F.normalize(z_t, p=2, dim=1)
        dist_sq = F.relu(2.0 - 2.0 * (z_s @ z_t.T))
        weights = F.relu(cosine_similarity(raw_s, raw_t) - sim_threshold)
        sic = (weights * dist_sq).sum() / (weights.sum() + 1e-8)

    mean_s = raw_s.mean(dim=0, keepdim=True) if raw_s.size(0) else torch.zeros(1, raw_embeddings.size(1), device=device)
    mean_t = raw_t.mean(dim=0, keepdim=True) if raw_t.size(0) else torch.zeros(1, raw_embeddings.size(1), device=device)
    delta = torch.norm(F.normalize(mean_s, dim=1) - F.normalize(mean_t, dim=1), p=2)
    omega = torch.tensor(float(lambda_g), device=device) * torch.exp(-float(gamma_g) * delta.detach())

    gen_loss = sic - beta_id * diversity
    return gen_loss, sic.detach(), diversity.detach(), omega
