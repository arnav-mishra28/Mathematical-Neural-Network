"""mnn.chaos.dynamics — Nonlinear systems and bifurcation analysis."""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Optional, List

class NonlinearSystem:
    """General nonlinear system dx/dt = f(t,x). Poincaré sections, stability."""
    def __init__(self, rhs: Callable, dim: int, name="System"):
        self.rhs=rhs; self.dim=dim; self.name=name
    def integrate(self, x0, t_span, dt=0.01, method="RK45"):
        t_eval=np.arange(t_span[0],t_span[1],dt)
        sol=solve_ivp(self.rhs,t_span,x0,t_eval=t_eval,method=method,rtol=1e-9,atol=1e-12)
        return sol.t, sol.y.T
    def poincare_section(self, x0, t_span, section_var=2, section_value=0., dt=0.001):
        _,traj=self.integrate(x0,t_span,dt); pts=[]
        for i in range(len(traj)-1):
            xi=traj[i,section_var]-section_value; xn=traj[i+1,section_var]-section_value
            if xi<0<xn:
                frac=-xi/(xn-xi); pts.append(traj[i]+frac*(traj[i+1]-traj[i]))
        return np.array(pts) if pts else np.empty((0,self.dim))
    def jacobian_numerical(self, x, h=1e-6):
        f0=np.array(self.rhs(0,x),dtype=float); J=np.zeros((self.dim,self.dim))
        for j in range(self.dim):
            xp=x.copy(); xp[j]+=h; xm=x.copy(); xm[j]-=h
            J[:,j]=(np.array(self.rhs(0,xp))-np.array(self.rhs(0,xm)))/(2*h)
        return J
    def stability_at(self, x_star):
        J=self.jacobian_numerical(x_star); evals,evecs=np.linalg.eig(J); re=np.real(evals)
        if all(re<0): stab="stable node/focus"
        elif all(re>0): stab="unstable node/focus"
        elif any(re<0) and any(re>0): stab="saddle"
        else: stab="center"
        return {"eigenvalues":evals,"eigenvectors":evecs,"stability":stab,"hyperbolic":not any(np.abs(re)<1e-10)}
    def phase_portrait_2d(self, x_range, y_range, n=20):
        xs=np.linspace(*x_range,n); ys=np.linspace(*y_range,n); X,Y=np.meshgrid(xs,ys)
        U=np.zeros_like(X); V=np.zeros_like(Y)
        for i in range(n):
            for j in range(n):
                s=np.zeros(self.dim); s[0]=X[i,j]; s[1]=Y[i,j]
                d=self.rhs(0,s); U[i,j]=d[0]; V[i,j]=d[1]
        return X,Y,U,V
    def __repr__(self): return f"NonlinearSystem(name={self.name},dim={self.dim})"

class BifurcationAnalyzer:
    @staticmethod
    def logistic_map_bifurcation(r_range=(2.5,4.0), n_r=1000, n_iter=1000, n_skip=200, x0=0.5):
        if n_iter <= 0 or n_r <= 0:
            return np.array([]), np.array([])
        burn_in=min(max(n_skip,0),max(n_iter-1,0)); keep_iters=max(n_iter-burn_in,1)
        r_vals=np.linspace(*r_range,n_r); rp,xp=[],[]
        for r in r_vals:
            x=x0
            for _ in range(burn_in): x=r*x*(1-x)
            for _ in range(keep_iters): x=r*x*(1-x); rp.append(r); xp.append(x)
        return np.array(rp), np.array(xp)
    @staticmethod
    def henon_map_bifurcation(a_range=(0.8,1.4), b=0.3, n_a=500, n_iter=500, n_skip=200):
        if n_iter <= 0 or n_a <= 0:
            return np.array([]), np.array([])
        burn_in=min(max(n_skip,0),max(n_iter-1,0)); keep_iters=max(n_iter-burn_in,1)
        a_vals=np.linspace(*a_range,n_a); ap,xp=[],[]
        for a in a_vals:
            x,y=0.,0.
            for _ in range(burn_in): x,y=1-a*x**2+y,b*x
            for _ in range(keep_iters): x,y=1-a*x**2+y,b*x; ap.append(a); xp.append(x)
        return np.array(ap), np.array(xp)
