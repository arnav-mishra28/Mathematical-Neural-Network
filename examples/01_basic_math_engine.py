"""Example 01 — Basic Math Engine"""
import sys; sys.path.insert(0,"..")
import numpy as np
from mnn.core.math_engine import MathEngine
eng = MathEngine()
f = eng.define_function("sin(x)*exp(-x**2/2)", ["x"])
g = eng.define_function("x**2 + y**2", ["x","y"])
print(f"f  = {f}")
print(f"f' = {eng.differentiate(f,'x')}")
print(f"∇g = {eng.gradient(g)}")
print(f"∇²g = {eng.hessian(g)}")
print(f"∫f dx [−3,3] = {eng.integrate_numeric(f,'x',(-3,3)):.6f}")
print(f"Taylor(f,6)  = {eng.taylor_series(f,'x',0,6)}")
print(f"LaTeX: {eng.to_latex(eng.differentiate(f,'x'))}")
print("[OK] Example 01")
