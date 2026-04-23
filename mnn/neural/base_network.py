"""mnn.neural.base_network — Core MNN neural architecture."""
from __future__ import annotations
import torch, torch.nn as nn, numpy as np
from typing import Dict, Callable

class MNNBlock(nn.Module):
    def __init__(self, width, activation="tanh", skip=True):
        super().__init__()
        acts={"tanh":torch.tanh,"sin":torch.sin,"relu":torch.relu,
              "gelu":nn.functional.gelu,"silu":nn.functional.silu,
              "swish":lambda x:x*torch.sigmoid(x),
              "mish":lambda x:x*torch.tanh(nn.functional.softplus(x))}
        self.l1=nn.Linear(width,width); self.l2=nn.Linear(width,width)
        self.act=acts[activation]; self.skip=skip; self.norm=nn.LayerNorm(width)
    def forward(self,x):
        h=self.act(self.l1(x)); h=self.norm(self.l2(h))
        return self.act(h+x) if self.skip else self.act(h)

class MNNNetwork(nn.Module):
    """Mathematical Neural Network — learns mathematical structure, not just patterns."""
    def __init__(self, input_dim, output_dim, width=128, depth=4, activation="tanh", use_residual=True):
        super().__init__()
        self.input_dim=input_dim; self.output_dim=output_dim; self.width=width; self.depth=depth
        self.input_layer=nn.Sequential(nn.Linear(input_dim,width),nn.LayerNorm(width))
        self.blocks=nn.ModuleList([MNNBlock(width,activation,use_residual) for _ in range(depth)])
        self.output_layer=nn.Linear(width,output_dim)
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x):
        h=self.input_layer(x)
        for b in self.blocks: h=b(h)
        return self.output_layer(h)
    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def predict_numpy(self,x):
        self.eval()
        with torch.no_grad(): return self.forward(torch.tensor(x,dtype=torch.float32)).numpy()
    def summary(self): return f"MNNNetwork(in={self.input_dim},out={self.output_dim},w={self.width},d={self.depth},params={self.count_parameters():,})"
    def __repr__(self): return self.summary()

class FourierEmbedding(nn.Module):
    """Random Fourier features: x → [sin(Bx), cos(Bx)]"""
    def __init__(self,input_dim,n_features=64,scale=1.0):
        super().__init__()
        self.register_buffer("B",torch.randn(input_dim,n_features)*scale)
        self.out_dim=2*n_features
    def forward(self,x): xB=x@self.B; return torch.cat([torch.sin(xB),torch.cos(xB)],dim=-1)

class FourierMNNNetwork(nn.Module):
    """MNN with Fourier feature embedding — ideal for periodic/oscillatory functions."""
    def __init__(self,input_dim,output_dim,n_fourier=64,width=128,depth=4,scale=1.0):
        super().__init__()
        self.embed=FourierEmbedding(input_dim,n_fourier,scale)
        self.network=MNNNetwork(self.embed.out_dim,output_dim,width,depth)
    def forward(self,x): return self.network(self.embed(x))
    def predict_numpy(self,x):
        self.eval()
        with torch.no_grad(): return self.forward(torch.tensor(x,dtype=torch.float32)).numpy()
    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
