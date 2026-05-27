"""mnn.embeddings.training — Training Objectives for Theorem Embeddings.

Part 7: Train embeddings using four objectives:
  1. Structural Similarity — related theorems nearby
  2. Proof Continuity — proof steps follow smooth paths
  3. Algebraic Consistency — preserve symmetry/group properties
  4. Topological Regularization — preserve manifold structure
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class StructuralSimilarityLoss(nn.Module):
    """Loss 1: Structurally related theorems should have high overlap.

    Uses contrastive loss: positive pairs (related) should be close,
    negative pairs (unrelated) should be far in Hilbert space.
    """
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, r1: torch.Tensor, i1: torch.Tensor,
                r2: torch.Tensor, i2: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """labels: 1 if related, 0 if unrelated."""
        # Complex inner product magnitude: |<T1|T2>|
        z1 = torch.complex(r1, i1)
        z2 = torch.complex(r2, i2)
        overlap = torch.abs((z1.conj() * z2).sum(dim=-1))

        # Related: maximize overlap (minimize 1 - overlap)
        pos_loss = labels * (1 - overlap) ** 2
        # Unrelated: push apart (minimize max(0, overlap - margin))
        neg_loss = (1 - labels) * F.relu(overlap - self.margin) ** 2

        return (pos_loss + neg_loss).mean()


class ProofContinuityLoss(nn.Module):
    """Loss 2: Consecutive proof steps should follow smooth paths.

    Minimizes total path length and curvature of proof trajectories.
    """
    def __init__(self, smoothness_weight: float = 0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight

    def forward(self, trajectory_r: torch.Tensor,
                trajectory_i: torch.Tensor) -> torch.Tensor:
        """trajectory: (n_steps, embed_dim)"""
        z = torch.complex(trajectory_r, trajectory_i)

        # Step distances: ||z_{t+1} - z_t||
        steps = z[1:] - z[:-1]
        step_lengths = torch.abs(steps).sum(dim=-1)

        # Path length loss
        path_loss = step_lengths.mean()

        # Smoothness: variance of step sizes
        if len(step_lengths) > 1:
            smoothness_loss = step_lengths.var()
        else:
            smoothness_loss = torch.tensor(0.0)

        return path_loss + self.smoothness_weight * smoothness_loss


class AlgebraicConsistencyLoss(nn.Module):
    """Loss 3: Preserve algebraic structure in embedding space.

    If op(A, B) = C algebraically, then embed(C) should be close
    to some combination of embed(A) and embed(B).
    """
    def __init__(self):
        super().__init__()

    def forward(self, a_r: torch.Tensor, a_i: torch.Tensor,
                b_r: torch.Tensor, b_i: torch.Tensor,
                c_r: torch.Tensor, c_i: torch.Tensor) -> torch.Tensor:
        """c = op(a, b) algebraically. Embed(c) ~ combine(embed(a), embed(b))."""
        # Use superposition as the algebraic combination
        combined_r = (a_r + b_r) / 2
        combined_i = (a_i + b_i) / 2
        # Normalize
        norm = torch.sqrt((combined_r**2 + combined_i**2).sum(dim=-1, keepdim=True) + 1e-8)
        combined_r = combined_r / norm
        combined_i = combined_i / norm

        # Overlap with actual result
        z_combined = torch.complex(combined_r, combined_i)
        z_c = torch.complex(c_r, c_i)
        overlap = torch.abs((z_combined.conj() * z_c).sum(dim=-1))

        return (1 - overlap).mean()


class TopologicalRegularizer(nn.Module):
    """Loss 4: Preserve local manifold structure.

    Nearby theorems in the original space should remain nearby in
    embedding space (topology preservation).
    """
    def __init__(self, n_neighbors: int = 5):
        super().__init__()
        self.n_neighbors = n_neighbors

    def forward(self, original_distances: torch.Tensor,
                embed_r: torch.Tensor, embed_i: torch.Tensor) -> torch.Tensor:
        """original_distances: (n, n) pairwise distances in original space.
        embed_r, embed_i: (n, dim) embeddings.
        """
        z = torch.complex(embed_r, embed_i)
        # Compute embedding distances
        G = z @ z.conj().T
        embed_dists = 1 - torch.abs(G)  # 0 = identical, 1 = orthogonal

        # Rank correlation: preserve ordering of distances
        n = len(embed_r)
        loss = torch.tensor(0.0, device=embed_r.device)
        count = 0
        for idx in range(min(n, 50)):  # sample for efficiency
            orig_row = original_distances[idx]
            embed_row = embed_dists[idx]
            # For each neighbor pair, check ordering
            _, orig_order = torch.sort(orig_row)
            neighbors = orig_order[:self.n_neighbors + 1]
            # Neighbors should have small embedding distances
            loss += embed_row[neighbors].mean()
            count += 1

        return loss / max(count, 1)


class TheoremEmbeddingTrainer:
    """End-to-end trainer for theorem embeddings with all 4 objectives."""

    def __init__(self, encoder: nn.Module, lr: float = 1e-3,
                 structural_weight: float = 1.0,
                 continuity_weight: float = 0.5,
                 algebraic_weight: float = 0.3,
                 topological_weight: float = 0.2):
        self.encoder = encoder
        self.optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
        self.structural_loss = StructuralSimilarityLoss()
        self.continuity_loss = ProofContinuityLoss()
        self.algebraic_loss = AlgebraicConsistencyLoss()
        self.topological_loss = TopologicalRegularizer()
        self.weights = {
            "structural": structural_weight,
            "continuity": continuity_weight,
            "algebraic": algebraic_weight,
            "topological": topological_weight,
        }
        self.history: List[Dict] = []

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One training step with available losses."""
        self.encoder.train()
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        losses = {}

        # Structural similarity (if pairs provided)
        if "pairs_1" in batch and "pairs_2" in batch:
            r1, i1 = self.encoder(batch["pairs_1"])
            r2, i2 = self.encoder(batch["pairs_2"])
            sl = self.structural_loss(r1, i1, r2, i2, batch["pair_labels"])
            total_loss = total_loss + self.weights["structural"] * sl
            losses["structural"] = sl.item()

        # Proof continuity (if trajectory provided)
        if "trajectory" in batch:
            traj = batch["trajectory"]
            tr, ti = self.encoder(traj)
            cl = self.continuity_loss(tr, ti)
            total_loss = total_loss + self.weights["continuity"] * cl
            losses["continuity"] = cl.item()

        # Algebraic consistency (if triples provided)
        if "alg_a" in batch and "alg_b" in batch and "alg_c" in batch:
            ar, ai = self.encoder(batch["alg_a"])
            br, bi = self.encoder(batch["alg_b"])
            cr, ci = self.encoder(batch["alg_c"])
            al = self.algebraic_loss(ar, ai, br, bi, cr, ci)
            total_loss = total_loss + self.weights["algebraic"] * al
            losses["algebraic"] = al.item()

        if total_loss.requires_grad:
            total_loss.backward()
            self.optimizer.step()

        losses["total"] = total_loss.item()
        self.history.append(losses)
        return losses

    def train_structural(self, pairs_1: torch.Tensor, pairs_2: torch.Tensor,
                          labels: torch.Tensor, n_epochs: int = 100,
                          verbose: bool = True) -> List[float]:
        """Train with structural similarity only."""
        losses = []
        for ep in range(n_epochs):
            batch = {"pairs_1": pairs_1, "pairs_2": pairs_2, "pair_labels": labels}
            result = self.train_step(batch)
            losses.append(result["total"])
            if verbose and (ep + 1) % max(1, n_epochs // 5) == 0:
                print(f"  Epoch {ep+1}/{n_epochs}: loss={result['total']:.6f}")
        return losses

    def embedding_report(self, token_ids: torch.Tensor,
                          names: List[str]) -> str:
        """Generate report on learned embeddings."""
        self.encoder.eval()
        with torch.no_grad():
            r, i = self.encoder(token_ids)
            r_np, i_np = r.numpy(), i.numpy()

        from .complex_embed import TheoremSimilarity
        sim_matrix = TheoremSimilarity.similarity_matrix(r_np, i_np)

        lines = ["Theorem Embedding Report", "=" * 40]
        for idx, name in enumerate(names):
            mag = np.sqrt(r_np[idx]**2 + i_np[idx]**2)
            lines.append(f"  {name}: dim={len(r_np[idx])}, "
                        f"mag_range=[{mag.min():.3f}, {mag.max():.3f}]")

        lines.append("\nSimilarity Matrix:")
        header = "        " + "  ".join(f"{n[:6]:>6}" for n in names)
        lines.append(header)
        for i_row, name in enumerate(names):
            row = f"{name[:6]:>6}  " + "  ".join(
                f"{sim_matrix[i_row, j]:.3f}" for j in range(len(names)))
            lines.append(row)

        return "\n".join(lines)
