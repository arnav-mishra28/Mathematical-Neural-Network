"""mnn.chaos.attractors — Strange attractor implementations."""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional, Dict

class BaseAttractor:
    def __init__(self, name, params): self.name=name; self.params=params
    def ode(self, t, s): raise NotImplementedError
    def _default_initial(self): return np.array([1.,1.,1.])
    def integrate(self, initial_state=None, t_span=(0,50), dt=0.01, method="RK45"):
        x0=initial_state if initial_state is not None else self._default_initial()
        t_eval=np.arange(t_span[0],t_span[1],dt)
        sol=solve_ivp(self.ode,t_span,x0,t_eval=t_eval,method=method,rtol=1e-9,atol=1e-12)
        return sol.y.T
    def __repr__(self): return f"{self.__class__.__name__}({self.params})"

class LorenzAttractor(BaseAttractor):
    """Lorenz (1963): dx/dt=σ(y-x), dy/dt=x(ρ-z)-y, dz/dt=xy-βz"""
    def __init__(self, sigma=10., rho=28., beta=8/3):
        super().__init__("Lorenz",{"σ":sigma,"ρ":rho,"β":beta})
        self.sigma=sigma; self.rho=rho; self.beta=beta
    def ode(self,t,s):
        x,y,z=s
        return [self.sigma*(y-x), x*(self.rho-z)-y, x*y-self.beta*z]
    def fixed_points(self):
        if self.rho<=1: return [np.zeros(3)]
        c=np.sqrt(self.beta*(self.rho-1))
        return [np.zeros(3),np.array([c,c,self.rho-1]),np.array([-c,-c,self.rho-1])]
    def jacobian_at(self, pt):
        x,y,z=pt
        return np.array([[-self.sigma,self.sigma,0],[self.rho-z,-1,-x],[y,x,-self.beta]])

class RosslerAttractor(BaseAttractor):
    """Rössler (1976): dx/dt=-y-z, dy/dt=x+ay, dz/dt=b+z(x-c)"""
    def __init__(self, a=.2, b=.2, c=5.7):
        super().__init__("Rössler",{"a":a,"b":b,"c":c}); self.a=a; self.b=b; self.c=c
    def ode(self,t,s):
        x,y,z=s; return [-y-z, x+self.a*y, self.b+z*(x-self.c)]
    def _default_initial(self): return np.array([1.,0.,0.])

class ChenAttractor(BaseAttractor):
    """Chen (1999): dx/dt=a(y-x), dy/dt=(c-a)x-xz+cy, dz/dt=xy-bz"""
    def __init__(self, a=35., b=3., c=28.):
        super().__init__("Chen",{"a":a,"b":b,"c":c}); self.a=a; self.b=b; self.c=c
    def ode(self,t,s):
        x,y,z=s; return [self.a*(y-x),(self.c-self.a)*x-x*z+self.c*y,x*y-self.b*z]
    def _default_initial(self): return np.array([-0.1,.5,-.6])

class HalvorsenAttractor(BaseAttractor):
    """Halvorsen: cyclic symmetry attractor."""
    def __init__(self, a=1.4):
        super().__init__("Halvorsen",{"a":a}); self.a=a
    def ode(self,t,s):
        x,y,z=s; a=self.a
        return [-a*x-4*y-4*z-y**2,-a*y-4*z-4*x-z**2,-a*z-4*x-4*y-x**2]
    def _default_initial(self): return np.array([-5.,0.,0.])

class ThomasAttractor(BaseAttractor):
    """Thomas' cyclically symmetric: dx/dt=sin(y)-bx, etc."""
    def __init__(self, b=0.208186):
        super().__init__("Thomas",{"b":b}); self.b=b
    def ode(self,t,s):
        x,y,z=s; return [np.sin(y)-self.b*x,np.sin(z)-self.b*y,np.sin(x)-self.b*z]
    def _default_initial(self): return np.array([.1,0.,0.])
