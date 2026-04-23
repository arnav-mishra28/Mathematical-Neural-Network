"""Example 02 — Neural Constraint Learning"""
import sys; sys.path.insert(0,"..")
import numpy as np
from mnn.neural.base_network     import MNNNetwork, FourierMNNNetwork
from mnn.neural.training         import MNNTrainer
from mnn.neural.constraint_layer import PDEConstraint
# Regression
x = np.linspace(-4,4,800).reshape(-1,1).astype(np.float32)
y = (np.sin(3*x)*np.exp(-x**2/4)).astype(np.float32)
net = MNNNetwork(1,1,64,4)
tr  = MNNTrainer(net,5e-3)
tr.train_supervised(x,y,500,verbose=False)
mse = float(np.mean((tr.evaluate(x)-y)**2))
print(f"Regression MSE = {mse:.6f}")
# PDE Laplace
net3 = MNNNetwork(2,1,32,3)
fn   = PDEConstraint.laplace_2d(net3)
xy   = np.random.uniform(0,1,(300,2)).astype(np.float32)
tr3  = MNNTrainer(net3,1e-3)
tk   = tr3.train_constrained(xy,fn,n_epochs=300,verbose=False)
print(f"Laplace PDE loss = {tk.latest().get('pde_loss',0):.6f}")
print("[OK] Example 02")
