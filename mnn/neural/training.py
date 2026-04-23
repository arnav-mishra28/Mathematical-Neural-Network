"""mnn.neural.training — Constraint-aware training loop for MNN."""
from __future__ import annotations
import torch, torch.nn as nn, numpy as np
from typing import Callable, Optional, Dict, List
from tqdm import tqdm

class LossTracker:
    def __init__(self): self.history: Dict[str,List[float]]={}
    def update(self,d):
        for k,v in d.items(): self.history.setdefault(k,[]).append(v)
    def latest(self): return {k:v[-1] for k,v in self.history.items()}
    def to_numpy(self): return {k:np.array(v) for k,v in self.history.items()}

class MNNTrainer:
    """Research-grade trainer: supervised + PDE-constrained + boundary conditions."""
    def __init__(self, network, lr=1e-3, optimizer="adam", device="cpu"):
        self.network=network.to(device); self.device=device; self.tracker=LossTracker()
        opt_map={"adam":torch.optim.Adam,"adamw":torch.optim.AdamW}
        if optimizer=="lbfgs":
            self.optimizer=torch.optim.LBFGS(network.parameters(),lr=lr,max_iter=20)
        else:
            self.optimizer=opt_map.get(optimizer,torch.optim.Adam)(network.parameters(),lr=lr)
        self.scheduler=None
    def _t(self,x): return (x if isinstance(x,torch.Tensor) else torch.tensor(np.array(x),dtype=torch.float32)).to(self.device)
    def train_supervised(self, x_train, y_train, n_epochs=1000, batch_size=None, verbose=True, print_every=100):
        X,Y=self._t(x_train),self._t(y_train); loss_fn=nn.MSELoss()
        it=tqdm(range(n_epochs),desc="Training") if verbose else range(n_epochs)
        for ep in it:
            if batch_size:
                idx=torch.randperm(X.shape[0])[:batch_size]; Xb,Yb=X[idx],Y[idx]
            else: Xb,Yb=X,Y
            self.optimizer.zero_grad()
            loss=loss_fn(self.network(Xb),Yb); loss.backward(); self.optimizer.step()
            if self.scheduler: self.scheduler.step()
            self.tracker.update({"data_loss":loss.item()})
            if verbose and (ep+1)%print_every==0: tqdm.write(f"[{ep+1}] loss={loss.item():.6f}")
        return self.tracker
    def train_constrained(self, collocation_pts, pde_fn, x_data=None, y_data=None,
                          boundary_loss_fn=None, w_pde=1.0, w_data=1.0, w_bc=1.0,
                          n_epochs=1000, verbose=True, print_every=100):
        Xc=self._t(collocation_pts); has_data=x_data is not None
        if has_data: Xd,Yd=self._t(x_data),self._t(y_data)
        it=tqdm(range(n_epochs),desc="MNN Training") if verbose else range(n_epochs)
        for ep in it:
            self.optimizer.zero_grad(); losses={}
            res=pde_fn(Xc); pde_l=w_pde*torch.mean(res**2)
            losses["pde_loss"]=pde_l.item(); total=pde_l
            if has_data:
                dl=w_data*nn.functional.mse_loss(self.network(Xd),Yd)
                losses["data_loss"]=dl.item(); total=total+dl
            if boundary_loss_fn:
                bl=w_bc*boundary_loss_fn(); losses["bc_loss"]=bl.item(); total=total+bl
            losses["total"]=total.item(); total.backward(); self.optimizer.step()
            if self.scheduler: self.scheduler.step()
            self.tracker.update(losses)
            if verbose and (ep+1)%print_every==0:
                tqdm.write(f"[{ep+1}] "+" | ".join(f"{k}={v:.5f}" for k,v in losses.items()))
        return self.tracker
    def evaluate(self, x) -> np.ndarray:
        self.network.eval()
        with torch.no_grad(): return self.network(self._t(x)).cpu().numpy()
    def save(self,path): torch.save(self.network.state_dict(),path)
    def load(self,path): self.network.load_state_dict(torch.load(path,map_location=self.device))
