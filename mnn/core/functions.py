"""mnn.core.functions — Special functions and transforms for MNN."""
from __future__ import annotations
import numpy as np
import sympy as sp
from scipy.special import gamma, erf, jv, yv, iv, kv
from scipy.fft import fft, ifft, fftfreq
from typing import Callable, List, Tuple, Dict

class MathFunctions:
    @staticmethod
    def gamma_function(x): return gamma(np.array(x,dtype=float))
    @staticmethod
    def beta_function(a,b): return gamma(a)*gamma(b)/gamma(a+b)
    @staticmethod
    def error_function(x): return erf(np.array(x,dtype=float))
    @staticmethod
    def bessel_j(n,x): return jv(n,np.array(x,dtype=float))
    @staticmethod
    def bessel_y(n,x): return yv(n,np.array(x,dtype=float))
    @staticmethod
    def bessel_i(n,x): return iv(n,np.array(x,dtype=float))
    @staticmethod
    def bessel_k(n,x): return kv(n,np.array(x,dtype=float))
    @staticmethod
    def legendre_polynomial(n,x):
        x_a=np.atleast_1d(np.array(x,dtype=float)); sym=sp.Symbol("x")
        return sp.lambdify(sym, sp.legendre(n,sym), modules=["numpy"])(x_a)
    @staticmethod
    def hermite_polynomial(n,x):
        x_a=np.atleast_1d(np.array(x,dtype=float)); sym=sp.Symbol("x")
        return sp.lambdify(sym, sp.hermite(n,sym), modules=["numpy"])(x_a)
    @staticmethod
    def chebyshev_polynomial(n,x):
        x_a=np.atleast_1d(np.array(x,dtype=float)); sym=sp.Symbol("x")
        return sp.lambdify(sym, sp.chebyshevt(n,sym), modules=["numpy"])(x_a)
    @staticmethod
    def fourier_transform(signal,dt=1.0):
        n=len(signal); return fftfreq(n,d=dt), fft(signal)
    @staticmethod
    def inverse_fourier_transform(spectrum): return np.real(ifft(spectrum))
    @staticmethod
    def power_spectrum(signal,dt=1.0):
        f,s=MathFunctions.fourier_transform(signal,dt); return f,np.abs(s)**2
    @staticmethod
    def laplace_transform(expr_str,t_var="t",s_var="s"):
        t,s=sp.symbols(f"{t_var} {s_var}")
        expr=sp.sympify(expr_str,locals={t_var:t,"exp":sp.exp,"sin":sp.sin,"cos":sp.cos})
        return sp.laplace_transform(expr,t,s)[0]
    @staticmethod
    def mexican_hat_wavelet(x,sigma=1.0):
        return (2/(np.sqrt(3*sigma)*np.pi**0.25))*(1-(x/sigma)**2)*np.exp(-x**2/(2*sigma**2))
    @staticmethod
    def morlet_wavelet(x,omega0=6.0):
        return np.pi**(-0.25)*np.exp(1j*omega0*x)*np.exp(-x**2/2)
    @staticmethod
    def inner_product(f,g,dx=1.0): return float(np.trapz(f*np.conj(g),dx=dx))
    @staticmethod
    def l2_norm(f,dx=1.0): return float(np.sqrt(np.trapz(np.abs(f)**2,dx=dx)))
    @staticmethod
    def gram_schmidt(basis,dx=1.0):
        ortho=[]
        for v in basis:
            w=v.copy().astype(complex)
            for u in ortho: w-=MathFunctions.inner_product(w,u,dx)*u
            n=MathFunctions.l2_norm(w,dx)
            if n>1e-12: ortho.append(w/n)
        return ortho
    @staticmethod
    def greens_function_1d(x,x0): return -np.abs(x-x0)/2.0
    @staticmethod
    def greens_function_3d(r): return 1.0/(4*np.pi*np.maximum(r,1e-15))
    def __repr__(self): return "MathFunctions()"
