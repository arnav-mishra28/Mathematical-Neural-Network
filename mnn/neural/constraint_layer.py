"""mnn.neural.constraint_layer — PDE and algebraic constraints via autograd."""
from __future__ import annotations
import torch, torch.nn as nn
from typing import Callable, Optional, Tuple

class ConstraintLayer(nn.Module):
    def __init__(self, network, constraint_fn=None, weight=1.0):
        super().__init__()
        self.network=network; self.constraint_fn=constraint_fn; self.weight=weight
    def forward(self,x):
        pred=self.network(x)
        res=self.constraint_fn(x,pred) if self.constraint_fn else torch.tensor(0.0)
        return pred, res
    def constraint_loss(self,x): _,r=self.forward(x); return self.weight*torch.mean(r**2)

class PDEConstraint:
    """PDE residual constraints via automatic differentiation."""
    @staticmethod
    def laplace_2d(net) -> Callable:
        def res(xy):
            xy=xy.requires_grad_(True); u=net(xy)
            ux=torch.autograd.grad(u.sum(),xy,create_graph=True)[0]
            uxx=torch.autograd.grad(ux[:,0].sum(),xy,create_graph=True)[0][:,0]
            uyy=torch.autograd.grad(ux[:,1].sum(),xy,create_graph=True)[0][:,1]
            return uxx+uyy
        return res
    @staticmethod
    def heat_1d(net, alpha=1.0) -> Callable:
        def res(xt):
            xt=xt.requires_grad_(True); u=net(xt)
            uxt=torch.autograd.grad(u.sum(),xt,create_graph=True)[0]
            ut=uxt[:,1]
            uxx=torch.autograd.grad(uxt[:,0].sum(),xt,create_graph=True)[0][:,0]
            return ut-alpha*uxx
        return res
    @staticmethod
    def wave_1d(net, c=1.0) -> Callable:
        def res(xt):
            xt=xt.requires_grad_(True); u=net(xt)
            uxt=torch.autograd.grad(u.sum(),xt,create_graph=True)[0]
            utt=torch.autograd.grad(uxt[:,1].sum(),xt,create_graph=True)[0][:,1]
            uxx=torch.autograd.grad(uxt[:,0].sum(),xt,create_graph=True)[0][:,0]
            return utt-c**2*uxx
        return res
    @staticmethod
    def burgers_1d(net, nu=0.01) -> Callable:
        def res(xt):
            xt=xt.requires_grad_(True); u=net(xt)
            uxt=torch.autograd.grad(u.sum(),xt,create_graph=True)[0]
            ut=uxt[:,1]; ux=uxt[:,0]
            uxx=torch.autograd.grad(ux.sum(),xt,create_graph=True)[0][:,0]
            return ut+u.squeeze()*ux-nu*uxx
        return res
    @staticmethod
    def schrodinger_1d(net, V_fn=None) -> Callable:
        def res(x, E=1.0):
            x=x.requires_grad_(True); psi=net(x)
            px=torch.autograd.grad(psi.sum(),x,create_graph=True)[0]
            pxx=torch.autograd.grad(px.sum(),x,create_graph=True)[0]
            V=V_fn(x) if V_fn else torch.zeros_like(x[:,0])
            return -0.5*pxx.squeeze()+V*psi.squeeze()-E*psi.squeeze()
        return res

class AlgebraicConstraint:
    @staticmethod
    def pythagorean(net): return lambda x: net(x)[:,0]**2+net(x)[:,1]**2-1.0
    @staticmethod
    def symmetry(net):    return lambda x: net(x)-net(-x)
    @staticmethod
    def antisymmetry(net):return lambda x: net(x)+net(-x)

class BoundaryCondition:
    @staticmethod
    def dirichlet(net, bp, bv):
        def loss(): return nn.functional.mse_loss(net(bp),bv)
        return loss
    @staticmethod
    def neumann(net, bp, nv, direction=0):
        def loss():
            b=bp.requires_grad_(True); u=net(b)
            ux=torch.autograd.grad(u.sum(),b,create_graph=True)[0]
            return nn.functional.mse_loss(ux[:,direction],nv)
        return loss
