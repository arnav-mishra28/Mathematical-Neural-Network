"""Tests for mnn — test_core"""
import sys; sys.path.insert(0,"..")
import numpy as np
from mnn.core.math_engine import MathEngine, ScalarField, VectorField, TensorObject
from mnn.core.operators   import DifferentialOperators
from mnn.core.functions   import MathFunctions
import sympy as sp
eng = MathEngine()
def test_scalar_field():
    f=eng.define_function("x**2",["x"]); assert abs(f.evaluate({"x":3.})-9.)<1e-9
def test_gradient():
    g=eng.define_function("x**2+y**2",["x","y"])
    assert abs(g.gradient().evaluate({"x":1.,"y":0.})[0]-2.)<1e-9
def test_laplacian():
    f=eng.define_function("x**2+y**2",["x","y"])
    assert abs(DifferentialOperators.laplacian(f).evaluate({"x":0.,"y":0.})-4.)<1e-9
def test_divergence():
    F=eng.define_vector_field(["x","y","z"],["x","y","z"])
    assert abs(DifferentialOperators.divergence(F).evaluate({"x":0,"y":0,"z":0})-3.)<1e-9
def test_curl_zero():
    f=eng.define_function("x**2+y**2+z**2",["x","y","z"])
    c=DifferentialOperators.curl(f.gradient()).evaluate({"x":1.,"y":2.,"z":3.})
    assert np.allclose(c,0,atol=1e-9)
def test_integration():
    f=eng.define_function("x**2",["x"])
    assert abs(eng.integrate_numeric(f,"x",(0,1))-1/3)<1e-5
def test_taylor():
    f=eng.define_function("exp(x)",["x"]); ts=eng.taylor_series(f,"x",0,4)
    x=sp.Symbol("x"); assert sp.simplify(ts-(1+x+x**2/2+x**3/6))==0
def test_bessel():
    j0=MathFunctions.bessel_j(0,np.array([0.])); assert abs(float(j0)-1.)<1e-6
def test_gram_schmidt():
    b=[np.array([1.,1.]),np.array([1.,0.])]
    o=MathFunctions.gram_schmidt(b,1.)
    assert abs(MathFunctions.inner_product(o[0],o[1]))<1e-10
if __name__=="__main__":
    for fn in [test_scalar_field,test_gradient,test_laplacian,test_divergence,
               test_curl_zero,test_integration,test_taylor,test_bessel,test_gram_schmidt]:
        fn(); print(f"  ✓ {fn.__name__}")
    print("[ALL CORE TESTS PASSED]")
