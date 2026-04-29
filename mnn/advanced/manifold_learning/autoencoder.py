"""
mnn.advanced.manifold_learning.autoencoder
============================================
Manifold-aware autoencoders for MNN.

Parts 2 & 3: Learn the manifold via compression/reconstruction,
then enforce topological constraints in the loss.

Classes
-------
  ManifoldAutoencoder  — standard AE with optional topology constraints
  ManifoldVAE          — variational AE for probabilistic manifold learning
  GeometricAE          — AE that preserves pairwise distances (isometric embedding)
  TopologicalAE        — AE with persistent homology regularisation

Key idea: the BOTTLENECK = latent manifold.
If reconstruction is good AND constraints are satisfied,
the network has discovered the intrinsic geometry.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
from tqdm import tqdm


# ── Shared building block ─────────────────────────────────────────────────────

def _mlp(dims: List[int], activation: str = "tanh",
          norm: bool = True) -> nn.Sequential:
    """Build an MLP from a list of layer widths."""
    acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    Act  = acts.get(activation, nn.Tanh)
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2:
            if norm: layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(Act())
    return nn.Sequential(*layers)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class AEResult:
    """Training result container for manifold autoencoders."""
    loss_history:    Dict[str, List[float]] = field(default_factory=dict)
    final_losses:    Dict[str, float]       = field(default_factory=dict)
    latent_dim:      int                    = 2
    ambient_dim:     int                    = 3
    n_epochs:        int                    = 0
    n_params:        int                    = 0
    manifold_name:   str                    = ""

    def summary(self) -> str:
        w = 54
        rows = [
            "╔" + "═"*w + "╗",
            f"║  Manifold Autoencoder: {self.manifold_name:<{w-24}}║",
            "╠" + "═"*w + "╣",
            f"║  Ambient dim  : {self.ambient_dim:<{w-17}}║",
            f"║  Latent dim   : {self.latent_dim:<{w-17}}║",
            f"║  Params       : {self.n_params:<{w-17},}║",
            f"║  Epochs       : {self.n_epochs:<{w-17}}║",
            "╠" + "═"*w + "╣",
        ]
        for k,v in self.final_losses.items():
            rows.append(f"║  {k:<22}: {v:<{w-26}.8f}║")
        rows.append("╚" + "═"*w + "╝")
        return "\n".join(rows)


# ── Standard Manifold Autoencoder ─────────────────────────────────────────────

class ManifoldAutoencoder(nn.Module):
    """
    Manifold Autoencoder: ambient_dim → latent_dim → ambient_dim.

    The encoder compresses points to the latent space (intrinsic manifold).
    The decoder reconstructs ambient-space points from latent codes.

    Topological constraints
    -----------------------
    The training loss includes:
      L_recon   = ‖x - decode(encode(x))‖²     (reconstruction)
      L_topo    = topological constraint on latent codes (optional)
      L_ambient = constraint on reconstructed points (e.g., on-sphere)

    The result: a trained encoder = coordinate chart, decoder = parametrisation.
    """

    def __init__(self, ambient_dim: int, latent_dim: int,
                 encoder_widths: List[int] = None,
                 decoder_widths: List[int] = None,
                 activation: str = "tanh"):
        super().__init__()
        self.ambient_dim = ambient_dim
        self.latent_dim  = latent_dim

        enc_dims = [ambient_dim] + (encoder_widths or [128, 64]) + [latent_dim]
        dec_dims = [latent_dim]  + (decoder_widths or [64, 128]) + [ambient_dim]

        self.encoder = _mlp(enc_dims, activation)
        self.decoder = _mlp(dec_dims, activation)

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, ambient) → z: (N, latent)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, latent) → x̂: (N, ambient)"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent_code)."""
        z    = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self.encode(torch.tensor(x, dtype=torch.float32)).numpy()

    def decode_numpy(self, z: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self.decode(torch.tensor(z, dtype=torch.float32)).numpy()

    def __repr__(self):
        return (f"ManifoldAutoencoder({self.ambient_dim}→{self.latent_dim}→{self.ambient_dim}, "
                f"params={self.count_parameters():,})")


# ── Variational Manifold Autoencoder ─────────────────────────────────────────

class ManifoldVAE(nn.Module):
    """
    Variational Autoencoder for probabilistic manifold learning.

    The encoder outputs μ and log σ² (mean and log-variance).
    The latent code is sampled: z = μ + σ·ε, ε ~ N(0,I).

    Loss = recon_loss + β · KL(q(z|x) ‖ p(z))
    where p(z) = N(0,I) is the prior.

    The β parameter controls the trade-off between reconstruction
    accuracy (β→0) and latent space regularity (β→1: standard VAE).
    """

    def __init__(self, ambient_dim: int, latent_dim: int,
                 hidden_widths: List[int] = None,
                 activation: str = "tanh",
                 beta: float = 1.0):
        super().__init__()
        self.ambient_dim = ambient_dim
        self.latent_dim  = latent_dim
        self.beta        = beta

        hid = hidden_widths or [128, 64]
        enc_base_dims = [ambient_dim] + hid
        self.enc_base = _mlp(enc_base_dims, activation)
        self.enc_mu   = nn.Linear(hid[-1], latent_dim)
        self.enc_logv = nn.Linear(hid[-1], latent_dim)

        dec_dims = [latent_dim] + hid[::-1] + [ambient_dim]
        self.decoder = _mlp(dec_dims, activation)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_var)."""
        h      = self.enc_base(x)
        mu     = self.enc_mu(h)
        log_var = self.enc_logv(h)
        return mu, log_var

    def reparametrize(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        """z = μ + σ·ε  (reparametrisation trick)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """Returns (reconstruction, mu, log_var, z_sample)."""
        mu, log_var = self.encode(x)
        z           = self.reparametrize(mu, log_var)
        x_hat       = self.decode(z)
        return x_hat, mu, log_var, z

    def kl_divergence(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        """KL(q ‖ N(0,I)) = -1/2 Σ(1 + log σ² - μ² - σ²)"""
        return -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())

    def vae_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_hat, mu, log_var, z = self.forward(x)
        recon = F.mse_loss(x_hat, x)
        kl    = self.kl_divergence(mu, log_var)
        total = recon + self.beta * kl
        return {"recon": recon, "kl": kl, "total": total, "z": z}

    def sample(self, n: int = 100) -> np.ndarray:
        """Sample n points from the learned manifold distribution."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim)
            return self.decode(z).numpy()

    def encode_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(torch.tensor(x, dtype=torch.float32))
            return mu.numpy()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"ManifoldVAE(ambient={self.ambient_dim}, latent={self.latent_dim}, "
                f"β={self.beta}, params={self.count_parameters():,})")


# ── Geometric AE (isometric / distance-preserving) ───────────────────────────

class GeometricAutoencoder(nn.Module):
    """
    Geometric Autoencoder that preserves pairwise distances.

    Loss = reconstruction + λ · distance_preservation
    where the distance term penalises distortion of the embedding:
      L_dist = ‖d_latent(zᵢ, zⱼ) - d_ambient(xᵢ, xⱼ)‖²  (for sampled pairs)

    This encourages the encoder to be an isometric embedding of the manifold.
    """

    def __init__(self, ambient_dim: int, latent_dim: int,
                 hidden_widths: List[int] = None,
                 activation: str = "tanh",
                 lambda_dist: float = 0.1):
        super().__init__()
        self.ambient_dim  = ambient_dim
        self.latent_dim   = latent_dim
        self.lambda_dist  = lambda_dist

        hid     = hidden_widths or [128, 64]
        enc_dims = [ambient_dim] + hid + [latent_dim]
        dec_dims = [latent_dim]  + hid[::-1] + [ambient_dim]
        self.encoder = _mlp(enc_dims, activation)
        self.decoder = _mlp(dec_dims, activation)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)

    def forward(self, x):
        z = self.encode(x); return self.decode(z), z

    def distance_preservation_loss(self, x: torch.Tensor,
                                    z: torch.Tensor,
                                    n_pairs: int = 200) -> torch.Tensor:
        """Sample random pairs and penalise latent/ambient distance mismatch."""
        N = x.shape[0]
        idx_i = torch.randint(0, N, (n_pairs,))
        idx_j = torch.randint(0, N, (n_pairs,))
        d_ambient = torch.norm(x[idx_i] - x[idx_j], dim=-1)
        d_latent  = torch.norm(z[idx_i] - z[idx_j], dim=-1)
        # Normalise to [0,1] range for scale-invariance
        d_amb_n = d_ambient / (d_ambient.max() + 1e-8)
        d_lat_n = d_latent  / (d_latent.max()  + 1e-8)
        return F.mse_loss(d_lat_n, d_amb_n)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_numpy(self, x):
        self.eval()
        with torch.no_grad():
            return self.encode(torch.tensor(x, dtype=torch.float32)).numpy()

    def __repr__(self):
        return (f"GeometricAutoencoder({self.ambient_dim}→{self.latent_dim}, "
                f"λ_dist={self.lambda_dist})")


# ── Manifold AE Trainer ───────────────────────────────────────────────────────

class ManifoldAETrainer:
    """
    Unified trainer for all manifold autoencoder types.
    Supports reconstruction loss + any combination of constraint losses.
    """

    def __init__(self, model: nn.Module,
                 lr: float = 1e-3,
                 device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device
        self.opt    = torch.optim.Adam(model.parameters(), lr=lr)
        self.sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=1000, eta_min=lr*0.01)
        self._extra_losses: List[Tuple[str, Callable, float]] = []
        self.result = AEResult(
            latent_dim=getattr(model, 'latent_dim', 2),
            ambient_dim=getattr(model, 'ambient_dim', 3),
            n_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
        )

    def add_constraint(self, name: str, loss_fn: Callable,
                       weight: float = 1.0) -> "ManifoldAETrainer":
        """Add an extra loss term: loss_fn(x_hat, z, x) → scalar."""
        self._extra_losses.append((name, loss_fn, weight))
        return self

    def train(self, data: np.ndarray,
              n_epochs: int = 2000,
              batch_size: int = 256,
              verbose: bool = True,
              print_every: int = 200,
              manifold_name: str = "") -> AEResult:
        X   = torch.tensor(data, dtype=torch.float32, device=self.device)
        N   = X.shape[0]
        itr = tqdm(range(n_epochs), desc=f"Manifold AE [{manifold_name}]") if verbose else range(n_epochs)

        for ep in itr:
            # Mini-batch
            idx  = torch.randperm(N)[:min(batch_size, N)]
            x_b  = X[idx]
            self.opt.zero_grad()
            losses = {}

            # Forward pass (handles both AE and VAE)
            if isinstance(self.model, ManifoldVAE):
                loss_dict = self.model.vae_loss(x_b)
                x_hat, z  = loss_dict["total"], loss_dict["z"]
                total      = loss_dict["total"]
                losses["recon"] = float(loss_dict["recon"].detach())
                losses["kl"]    = float(loss_dict["kl"].detach())
            else:
                x_hat, z  = self.model(x_b)
                recon      = F.mse_loss(x_hat, x_b)
                total      = recon
                losses["recon"] = float(recon.detach())

                # Geometric AE: add distance preservation
                if isinstance(self.model, GeometricAutoencoder):
                    d_loss = self.model.lambda_dist * self.model.distance_preservation_loss(x_b, z)
                    total  = total + d_loss
                    losses["dist"] = float(d_loss.detach())

            # Extra constraint losses
            for cname, cfn, cw in self._extra_losses:
                cl    = cw * cfn(x_hat, z, x_b)
                total = total + cl
                losses[cname] = float(cl.detach())

            losses["total"] = float(total.detach())
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step(); self.sched.step()

            for k, v in losses.items():
                self.result.loss_history.setdefault(k, []).append(v)

            if verbose and (ep+1) % print_every == 0:
                ls = "  ".join(f"{k}={v:.5f}" for k,v in losses.items())
                tqdm.write(f"  [{ep+1:>5}]  {ls}")

        self.result.n_epochs     = n_epochs
        self.result.final_losses = {k: v[-1] for k,v in self.result.loss_history.items()}
        self.result.manifold_name = manifold_name
        return self.result

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.model.encode_numpy(x)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            xt   = torch.tensor(x, dtype=torch.float32, device=self.device)
            xhat = self.model(xt)
            if isinstance(xhat, tuple): xhat = xhat[0]
            return xhat.cpu().numpy()

    def reconstruction_error(self, x: np.ndarray) -> float:
        x_hat = self.reconstruct(x)
        return float(np.mean((x_hat - x)**2))
