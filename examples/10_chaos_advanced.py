"""
Example 10 — Advanced Chaos Theory
=====================================
Fractal analysis, multifractals, chaotic maps,
coupled oscillators, Mandelbrot/Julia sets.
"""
import sys; sys.path.insert(0, "..")
import numpy as np
from mnn.chaos.attractors      import LorenzAttractor
from mnn.advanced.chaos_advanced import (
    FractalAnalyzer, MultifractalSpectrum, ChaoticMap, CoupledOscillators
)

print("=" * 56)
print("  MNN Example 10 — Advanced Chaos Theory")
print("=" * 56)

# ── 1. Box-counting dimension ─────────────────────────────────
print("\n[1] Fractal dimensions of Lorenz attractor")
lor  = LorenzAttractor()
traj = lor.integrate(t_span=(0, 40), dt=0.01)
d_box, sc, ct = FractalAnalyzer.box_counting_dimension(traj[::5], n_scales=12)
print(f"    Box-counting dim D₀ ≈ {d_box:.3f}  (expected ~2.05)")

# ── 2. Hurst exponents ────────────────────────────────────────
print("\n[2] Hurst exponent of Lorenz x-signal")
signal = traj[:, 0]
H_rs  = FractalAnalyzer.hurst_exponent_rs(signal[:3000])
H_dfa = FractalAnalyzer.hurst_exponent_dfa(signal[:3000])
print(f"    H (R/S)  = {H_rs:.4f}")
print(f"    H (DFA)  = {H_dfa:.4f}  (>0.5 → persistent)")

# ── 3. Lacunarity ─────────────────────────────────────────────
print("\n[3] Lacunarity of Lorenz attractor")
lac = FractalAnalyzer.lacunarity(traj[::10, :2], scales=8)
print(f"    Lacunarity Λ ≈ {lac:.4f}")

# ── 4. Multifractal spectrum ──────────────────────────────────
print("\n[4] Multifractal spectrum D_q")
mf    = MultifractalSpectrum(traj[::10])
qs, Dq = mf.generalized_dimensions(q_range=(-3,3), n_q=13, n_scales=8)
print(f"    D₀ ≈ {Dq[qs==min(qs, key=lambda x: abs(x))][0]:.3f} (capacity dim)")
print(f"    D_q range: [{Dq.min():.3f}, {Dq.max():.3f}]")
alpha, f_a = mf.singularity_spectrum(q_range=(-3,3), n_q=13)
print(f"    α range: [{alpha.min():.3f}, {alpha.max():.3f}]")

# ── 5. IFS fractals ───────────────────────────────────────────
print("\n[5] IFS fractals")
sier  = FractalAnalyzer.sierpinski_triangle(n_points=10000)
d_s, _, _ = FractalAnalyzer.box_counting_dimension(sier, n_scales=10)
print(f"    Sierpiński triangle dim ≈ {d_s:.3f}  (exact: {np.log(3)/np.log(2):.4f})")
fern  = FractalAnalyzer.barnsley_fern(n_points=10000)
d_f, _, _ = FractalAnalyzer.box_counting_dimension(fern, n_scales=10)
print(f"    Barnsley fern dim ≈ {d_f:.3f}")

# ── 6. Mandelbrot / Julia sets ────────────────────────────────
print("\n[6] Mandelbrot & Julia sets")
M = FractalAnalyzer.mandelbrot_set(n_real=100, n_imag=100, max_iter=50)
print(f"    Mandelbrot grid: {M.shape}, max iter used: {M.max()}")
J = FractalAnalyzer.julia_set(c=-0.7+0.27j, n_real=100, n_imag=100, max_iter=50)
print(f"    Julia set grid:  {J.shape}")

# ── 7. Chaotic maps ───────────────────────────────────────────
print("\n[7] Chaotic maps")
sm   = ChaoticMap.standard_map(K=1.5, n_iter=2000)
print(f"    Standard map (K=1.5): {sm.shape} pts")
cat  = ChaoticMap.arnolds_cat_map(n_iter=100)
print(f"    Arnold's cat map:     {cat.shape} pts")
ik   = ChaoticMap.ikeda_map(u=0.9, n_iter=3000)
print(f"    Ikeda map:            {ik.shape} pts")
tent = ChaoticMap.tent_map(n_iter=2000)
print(f"    Tent map signal:      {tent.shape}")
circ = ChaoticMap.circle_map(K=1.5, n_iter=2000)
print(f"    Circle map (K=1.5):   {circ.shape}")

# Lorenz return map
xn, xn1 = ChaoticMap.lorenz_return_map(traj)
print(f"    Lorenz return map:    {len(xn)} maxima")

# ── 8. Coupled oscillators ────────────────────────────────────
print("\n[8] Coupled oscillators")
t_arr, theta = CoupledOscillators.kuramoto_model(N=8, K=3.0, t_end=30.)
r = CoupledOscillators.order_parameter(theta)
print(f"    Kuramoto (N=8, K=3):  r_final = {r[-1]:.4f}  (→1 = synchronized)")
vdp  = CoupledOscillators.van_der_pol(mu=2., t_end=20.)
print(f"    Van der Pol:          traj shape={vdp.shape}")
duff = CoupledOscillators.duffing_oscillator(t_end=40.)
print(f"    Duffing oscillator:   traj shape={duff.shape}")

print("\n[OK] Example 10 complete.")
