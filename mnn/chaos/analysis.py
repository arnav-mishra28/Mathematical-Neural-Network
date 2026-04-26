"""mnn.chaos.analysis — Lyapunov exponents, fractal dimension, entropy, 0-1 test."""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Optional


def _normalized_autocorrelation(signal):
    sig=np.asarray(signal,dtype=float).ravel()
    if sig.size<2: return np.array([1.0])
    sig=sig-sig.mean()
    acf=np.correlate(sig,sig,mode="full")[sig.size-1:]
    if acf.size==0 or abs(acf[0])<1e-15: return np.array([1.0])
    return acf/acf[0]


def _decorrelation_lag(signal, threshold, max_lag):
    acf=_normalized_autocorrelation(signal)
    upper=min(max_lag,acf.size-1)
    for lag in range(1,upper+1):
        if acf[lag]<threshold: return lag
    return 1

class ChaosAnalyzer:
    @staticmethod
    def largest_lyapunov_exponent(trajectory, dt=0.01, min_sep=10):
        n,d=trajectory.shape; divs=[]
        dists=np.array([[np.linalg.norm(trajectory[i]-trajectory[j]) for j in range(n)] for i in range(n)])
        for i in range(n//2):
            row=dists[i].copy(); row[max(0,i-min_sep):i+min_sep]=np.inf
            j=np.argmin(row)
            if row[j]==np.inf or row[j]<1e-15: continue
            steps=min(n-i,n-j,n//4)
            if steps<2: continue
            dt_=np.linalg.norm(trajectory[i:i+steps]-trajectory[j:j+steps],axis=1)
            dt_=np.maximum(dt_,1e-15)
            divs.append(np.log(dt_[-1]/dt_[0])/(steps*dt))
        return float(np.mean(divs)) if divs else 0.0
    @staticmethod
    def lyapunov_spectrum(ode_fn, x0, t_end=100., dt=0.1, renorm_steps=10):
        n=len(x0); x=x0.copy().astype(float); Q=np.eye(n); exponents=np.zeros(n); count=0; t=0.
        while t<t_end:
            def aug(t_,s):
                x_=s[:n]; Q_=s[n:].reshape(n,n)
                dx=np.array(ode_fn(t_,x_),dtype=float); h=1e-7
                J=np.zeros((n,n))
                for j in range(n):
                    xp=x_.copy(); xp[j]+=h; xm=x_.copy(); xm[j]-=h
                    J[:,j]=(np.array(ode_fn(t_,xp))-np.array(ode_fn(t_,xm)))/(2*h)
                return np.concatenate([dx,(J@Q_).flatten()])
            sol=solve_ivp(aug,[t,t+dt*renorm_steps],np.concatenate([x,Q.flatten()]),method="RK45",rtol=1e-8,atol=1e-10,max_step=dt)
            x=sol.y[:n,-1]; Q=sol.y[n:,-1].reshape(n,n)
            Q,R=np.linalg.qr(Q); exponents+=np.log(np.abs(np.diag(R)))
            t+=dt*renorm_steps; count+=1
        return exponents/(count*dt*renorm_steps)
    @staticmethod
    def correlation_dimension(trajectory, n_r=30):
        n=len(trajectory)
        dists=np.array([np.linalg.norm(trajectory[i]-trajectory[j]) for i in range(n) for j in range(i+1,n)])
        r_min=np.percentile(dists,1); r_max=np.percentile(dists,50)
        r_vals=np.logspace(np.log10(r_min),np.log10(r_max),n_r)
        C_r=np.array([np.sum(dists<r)/len(dists) for r in r_vals])
        valid=C_r>0
        if valid.sum()<3: return 0.,r_vals,C_r
        slope,_=np.polyfit(np.log(r_vals[valid]),np.log(C_r[valid]),1)
        return float(slope),r_vals,C_r
    @staticmethod
    def kaplan_yorke_dimension(spectrum):
        ls=np.sort(spectrum)[::-1]; cs=np.cumsum(ls); j_arr=np.where(cs>=0)[0]
        if len(j_arr)==0: return 0.
        j=j_arr[-1]
        if j+1>=len(ls) or abs(ls[j+1])<1e-15: return float(j+1)
        return float(j+1)+cs[j]/abs(ls[j+1])
    @staticmethod
    def sample_entropy(signal, m=2, r=0.2):
        r_abs=r*np.std(signal); n=len(signal)
        def count(m_):
            c=0
            for i in range(n-m_):
                for j in range(i+1,n-m_):
                    if np.max(np.abs(signal[i:i+m_]-signal[j:j+m_]))<r_abs: c+=1
            return c
        A=count(m+1); B=count(m)
        return -np.log(A/B) if A>0 and B>0 else 0.
    @staticmethod
    def permutation_entropy(signal, order=3, delay=1):
        from math import factorial
        sig=np.asarray(signal,dtype=float).ravel()
        if order<2 or sig.size<order: return 0.0
        eff_delay=max(int(delay),1)
        if eff_delay==1 and sig.size>=order*8:
            eff_delay=_decorrelation_lag(sig,0.9,max(1,min(sig.size//(order+1),32)))
        n=len(sig); counts={}; total=0
        for i in range(n-(order-1)*eff_delay):
            sub=sig[i:i+order*eff_delay:eff_delay]; perm=tuple(np.argsort(sub,kind="mergesort"))
            counts[perm]=counts.get(perm,0)+1; total+=1
        if total==0: return 0.0
        probs=np.array(list(counts.values()))/total
        return -np.sum(probs*np.log2(probs+1e-15))/np.log2(factorial(order))
    @staticmethod
    def recurrence_matrix(trajectory, epsilon=None):
        n=len(trajectory)
        dists=np.array([[np.linalg.norm(trajectory[i]-trajectory[j]) for j in range(n)] for i in range(n)])
        eps=epsilon or np.percentile(dists,10); return (dists<eps).astype(int)
    @staticmethod
    def recurrence_quantification(R):
        n=R.shape[0]; RR=(R.sum()-n)/(n*(n-1)+1e-15)
        diag_lens=[]
        for k in range(1,n):
            d=np.diag(R,k); i=0
            while i<len(d):
                if d[i]==1:
                    j=i
                    while j<len(d) and d[j]==1: j+=1
                    diag_lens.append(j-i); i=j
                else: i+=1
        dl=np.array(diag_lens) if diag_lens else np.array([0])
        DET=np.sum(dl[dl>=2])/max(R.sum()-n,1); L_max=int(dl.max())
        return {"RR":float(RR),"DET":float(DET),"L_max":L_max}
    @staticmethod
    def zero_one_test(signal, n_samples=100):
        sig=np.asarray(signal,dtype=float).ravel()
        if sig.size<10: return 0.0
        if sig.size>=100:
            step=_decorrelation_lag(sig,0.2,max(1,min(sig.size//10,64)))
            sig=sig[::step]
        n=len(sig); rng=np.random.default_rng(42)
        Ks=[]
        for c in rng.uniform(np.pi/5,4*np.pi/5,n_samples):
            p=np.zeros(n); q=np.zeros(n)
            for j in range(1,n): p[j]=p[j-1]+sig[j-1]*np.cos(j*c); q[j]=q[j-1]+sig[j-1]*np.sin(j*c)
            N2=n//2
            if N2<=2: continue
            M=np.array([np.mean((p[s:s+N2]-p[:N2])**2+(q[s:s+N2]-q[:N2])**2) for s in range(1,N2)])
            if M.size==0 or np.std(M)<1e-12: continue
            K=np.corrcoef(np.arange(1,N2),M)[0,1]
            if not np.isnan(K): Ks.append(K)
        if not Ks: return 0.0
        return float(np.clip(np.median(Ks),0.0,1.0))
    def __repr__(self): return "ChaosAnalyzer()"
