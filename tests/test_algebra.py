"""Tests for mnn — test_algebra"""
import sys; sys.path.insert(0,"..")
import numpy as np
from mnn.algebra.groups  import Group, LieGroup
from mnn.algebra.abelian import AbelianGroup, AbelianFunction
from mnn.algebra.tensors import TensorEngine
from mnn.core.math_engine import TensorObject
def test_cyclic():       assert Group.cyclic(6).order==6
def test_abelian():      assert Group.cyclic(5).is_abelian()
def test_s3_nonab():     assert not Group.symmetric(3).is_abelian()
def test_dihedral():     assert Group.dihedral(4).order==8
def test_identity():     assert Group.cyclic(5).identity()==0
def test_inverse():      assert Group.cyclic(6).inverse(2)==4
def test_el_order():     assert Group.cyclic(6).element_order(2)==3
def test_fourier_roundtrip():
    Z8=AbelianGroup.cyclic(8); f={k:float(np.sin(k)) for k in range(8)}
    fr=Z8.inverse_fourier_transform(Z8.fourier_transform(f))
    assert all(abs(fr[k]-f[k])<1e-10 for k in range(8))
def test_theta3():       assert abs(AbelianFunction.theta3(0,1j)-1.0864348112133)<0.01
def test_levi_civita():
    e=TensorEngine.levi_civita(3).to_numpy()
    assert e[0,1,2]==1 and e[1,0,2]==-1 and e[0,0,1]==0
def test_einsum():
    T=TensorObject(np.eye(3)); T2=TensorObject(np.ones((3,3)))
    np.testing.assert_array_almost_equal(TensorEngine.einsum('ij,jk->ik',T,T2).to_numpy(),np.ones((3,3)))
def test_antisymmetrize():
    A=TensorObject(np.array([[1,2],[3,4]],dtype=float))
    r=TensorEngine.antisymmetrize(A).to_numpy()
    assert abs(r[0,0])<1e-10 and abs(r[0,1]+r[1,0])<1e-10
def test_so3_semisimple(): assert LieGroup.SO(3).is_semisimple()
def test_killing_nondegenerate():
    K=LieGroup.SO(3).killing_form(); assert abs(np.linalg.det(K))>1e-10
if __name__=="__main__":
    for fn in [test_cyclic,test_abelian,test_s3_nonab,test_dihedral,test_identity,test_inverse,
               test_el_order,test_fourier_roundtrip,test_theta3,test_levi_civita,test_einsum,
               test_antisymmetrize,test_so3_semisimple,test_killing_nondegenerate]:
        fn(); print(f"  ✓ {fn.__name__}")
    print("[ALL ALGEBRA TESTS PASSED]")
