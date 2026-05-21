"""mnn.quantum.complex_nn — Complex-valued neural networks.

Part 2: Neural layers that process complex numbers natively.
Instead of only scalar activations, we process magnitude AND phase.
This enables interference-like behavior and richer representations.
"""
from __future__ import annotations
import torch, torch.nn as nn, numpy as np
from typing import Optional, List, Dict, Callable


class ComplexLinear(nn.Module):
    """Complex-valued linear layer: W_c z + b_c where W_c, z, b_c in C^n.

    Parameterized as two real matrices: W_real, W_imag.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_real = nn.Linear(in_features, out_features, bias=bias)
        self.W_imag = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        scale = 1.0 / np.sqrt(self.in_features)
        nn.init.uniform_(self.W_real.weight, -scale, scale)
        nn.init.uniform_(self.W_imag.weight, -scale, scale)

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor):
        """(a+bi)(c+di) = (ac-bd) + (ad+bc)i"""
        out_real = self.W_real(z_real) - self.W_imag(z_imag)
        out_imag = self.W_real(z_imag) + self.W_imag(z_real)
        return out_real, out_imag


class ComplexActivation(nn.Module):
    """Phase-aware activation applied independently to magnitude and phase.

    Supports: 'modrelu', 'zrelu', 'cardioid', 'split_tanh'.
    """
    def __init__(self, kind: str = "modrelu", in_features: int = 1):
        super().__init__()
        self.kind = kind
        if kind == "modrelu":
            self.bias = nn.Parameter(torch.zeros(in_features))

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        if self.kind == "modrelu":
            mag = torch.sqrt(real**2 + imag**2 + 1e-8)
            phase_r = real / mag
            phase_i = imag / mag
            activated_mag = torch.relu(mag + self.bias)
            return activated_mag * phase_r, activated_mag * phase_i
        elif self.kind == "zrelu":
            mask = (real >= 0) & (imag >= 0)
            return real * mask.float(), imag * mask.float()
        elif self.kind == "cardioid":
            phase = torch.atan2(imag, real)
            scale = 0.5 * (1 + torch.cos(phase))
            return scale * real, scale * imag
        elif self.kind == "split_tanh":
            return torch.tanh(real), torch.tanh(imag)
        else:
            return real, imag


class ComplexLayerNorm(nn.Module):
    """Layer normalization for complex tensors."""
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta_r = nn.Parameter(torch.zeros(features))
        self.beta_i = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        mag = torch.sqrt(real**2 + imag**2 + self.eps)
        mean = mag.mean(dim=-1, keepdim=True)
        std = mag.std(dim=-1, keepdim=True) + self.eps
        scale = self.gamma / (std + self.eps)
        norm_factor = scale / (mag + self.eps)
        return real * norm_factor + self.beta_r, imag * norm_factor + self.beta_i


class ComplexDropout(nn.Module):
    """Dropout that drops entire complex values (both real and imaginary)."""
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        if self.training and self.p > 0:
            mask = torch.bernoulli(torch.full_like(real, 1 - self.p)) / (1 - self.p)
            return real * mask, imag * mask
        return real, imag


class ComplexBlock(nn.Module):
    """Residual block for complex networks: linear → activation → norm."""
    def __init__(self, width: int, activation: str = "modrelu",
                 dropout: float = 0.0, skip: bool = True):
        super().__init__()
        self.linear = ComplexLinear(width, width)
        self.act = ComplexActivation(activation, width)
        self.norm = ComplexLayerNorm(width)
        self.drop = ComplexDropout(dropout) if dropout > 0 else None
        self.skip = skip

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        r, i = self.linear(real, imag)
        r, i = self.act(r, i)
        r, i = self.norm(r, i)
        if self.drop:
            r, i = self.drop(r, i)
        if self.skip:
            r, i = r + real, i + imag
        return r, i


class ComplexNeuralNetwork(nn.Module):
    """Full complex-valued neural network: real inputs → complex processing → real outputs.

    Architecture: encode → N complex blocks → decode
    """
    def __init__(self, input_dim: int, output_dim: int,
                 width: int = 64, depth: int = 4,
                 activation: str = "modrelu", dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.encoder_real = nn.Linear(input_dim, width)
        self.encoder_imag = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([
            ComplexBlock(width, activation, dropout) for _ in range(depth)
        ])
        self.decoder = nn.Linear(2 * width, output_dim)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        r = self.encoder_real(x)
        i = self.encoder_imag(x)
        for block in self.blocks:
            r, i = block(r, i)
        combined = torch.cat([r, i], dim=-1)
        return self.decoder(combined)

    def complex_features(self, x: torch.Tensor):
        """Return complex representation (for analysis)."""
        r = self.encoder_real(x)
        i = self.encoder_imag(x)
        for block in self.blocks:
            r, i = block(r, i)
        return r, i

    def magnitude_phase(self, x: torch.Tensor):
        r, i = self.complex_features(x)
        mag = torch.sqrt(r**2 + i**2 + 1e-8)
        phase = torch.atan2(i, r)
        return mag, phase

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self(torch.tensor(x, dtype=torch.float32)).numpy()

    def __repr__(self):
        return (f"ComplexNeuralNetwork(in={self.input_dim}, out={self.output_dim}, "
                f"w={self.width}, d={self.depth}, params={self.count_parameters():,})")


class ComplexTrainer:
    """Trainer for complex-valued networks."""
    def __init__(self, model: ComplexNeuralNetwork, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.history: Dict[str, List[float]] = {"loss": []}

    def train(self, x: np.ndarray, y: np.ndarray,
              n_epochs: int = 1000, batch_size: int = 256,
              verbose: bool = True, print_every: int = 200) -> Dict:
        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
        loss_fn = nn.MSELoss()
        n = len(X)

        for ep in range(n_epochs):
            self.model.train()
            idx = torch.randperm(n)[:batch_size]
            pred = self.model(X[idx])
            loss = loss_fn(pred, Y[idx])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.history["loss"].append(loss.item())
            if verbose and (ep + 1) % print_every == 0:
                print(f"  [Complex] Epoch {ep+1}/{n_epochs} loss={loss.item():.6f}")

        return self.history
