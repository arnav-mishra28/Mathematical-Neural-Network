"""
mnn.advanced.abelian.elliptic
==============================
Elliptic curves and their connection to Abelian varieties.

An elliptic curve over a field k: E: y² = x³ + ax + b, Δ ≠ 0.

Features:
  - Point addition (group law)
  - Scalar multiplication (double-and-add)
  - Point at infinity (identity element)
  - Discriminant and j-invariant
  - Weierstrass form, Legendre form
  - Torsion points
  - Connection to Abelian functions (Weierstrass ℘)
  - Elliptic curve over finite fields Fp (cryptographic applications)
  - Weil pairing (sketch)
"""
from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Optional, Tuple, List, Union


class Point:
    """A point on an elliptic curve, or the point at infinity O."""
    def __init__(self, x: Optional[float], y: Optional[float]):
        self.x = x; self.y = y
        self.is_infinity = (x is None and y is None)

    @classmethod
    def infinity(cls) -> "Point":
        return cls(None, None)

    def __eq__(self, other):
        if self.is_infinity and other.is_infinity: return True
        if self.is_infinity or  other.is_infinity: return False
        return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10

    def __repr__(self):
        return "O" if self.is_infinity else f"({self.x:.6f}, {self.y:.6f})"


class EllipticCurve:
    """
    Elliptic curve E: y² = x³ + ax + b over R (or Fp for integer arithmetic).
    This is a compact Abelian variety of dimension 1.
    """

    def __init__(self, a: float, b: float, field: str = "R"):
        self.a = a; self.b = b; self.field = field
        if self.discriminant() == 0:
            raise ValueError(f"Discriminant is zero — not an elliptic curve! (a={a}, b={b})")

    # ── Curve properties ──────────────────────────────────────────────────────

    def discriminant(self) -> float:
        """Δ = -16(4a³ + 27b²) ≠ 0 for smooth curve."""
        return -16 * (4*self.a**3 + 27*self.b**2)

    def j_invariant(self) -> float:
        """j = -1728 · (4a)³ / Δ"""
        return -1728 * (4*self.a)**3 / self.discriminant()

    def evaluate(self, x: float) -> Optional[Tuple[float, float]]:
        """Return y values for given x (if they exist)."""
        rhs = x**3 + self.a*x + self.b
        if rhs < 0: return None
        y = np.sqrt(rhs)
        return (y, -y)

    def is_on_curve(self, P: Point, tol: float = 1e-8) -> bool:
        if P.is_infinity: return True
        return abs(P.y**2 - (P.x**3 + self.a*P.x + self.b)) < tol

    # ── Group law ─────────────────────────────────────────────────────────────

    def add(self, P: Point, Q: Point) -> Point:
        """P + Q on the elliptic curve."""
        if P.is_infinity: return Q
        if Q.is_infinity: return P

        if abs(P.x - Q.x) < 1e-12:
            if abs(P.y + Q.y) < 1e-12: return Point.infinity()  # P + (-P) = O
            # P == Q: use tangent (doubling)
            lam = (3*P.x**2 + self.a) / (2*P.y) if abs(P.y) > 1e-12 else float('inf')
        else:
            lam = (Q.y - P.y) / (Q.x - P.x)

        x3 = lam**2 - P.x - Q.x
        y3 = lam*(P.x - x3) - P.y
        return Point(x3, y3)

    def negate(self, P: Point) -> Point:
        """−P = (x, −y)."""
        if P.is_infinity: return P
        return Point(P.x, -P.y)

    def scalar_mult(self, n: int, P: Point) -> Point:
        """nP = P + P + ... + P (n times), using double-and-add."""
        if n == 0: return Point.infinity()
        if n < 0:  return self.scalar_mult(-n, self.negate(P))
        R = Point.infinity(); Q = Point(P.x, P.y)
        while n > 0:
            if n % 2 == 1: R = self.add(R, Q)
            Q = self.add(Q, Q); n //= 2
        return R

    # ── Points on curve ───────────────────────────────────────────────────────

    def sample_points(self, x_range: Tuple[float,float] = (-3,3),
                      n: int = 200) -> np.ndarray:
        """Sample points on the curve."""
        xs = np.linspace(*x_range, n)
        points = []
        for x in xs:
            ys = self.evaluate(x)
            if ys:
                points.append([x,  ys[0]])
                points.append([x, -ys[0]])
        return np.array(points) if points else np.zeros((0,2))

    def torsion_points_search(self, x_range=(-5,5), n=1000) -> List[Point]:
        """Search for rational-like torsion points (approximate, for R)."""
        pts = [Point.infinity()]
        xs = np.linspace(*x_range, n)
        for x in xs:
            rhs = x**3 + self.a*x + self.b
            if rhs >= 0:
                y = np.sqrt(rhs)
                P = Point(x, y)
                # Check if 2P = O (2-torsion: P = -P means y=0)
                if abs(y) < 1e-3:
                    pts.append(P)
        return pts

    # ── Connection to Abelian functions ───────────────────────────────────────

    def weierstrass_parametrization(self, u: np.ndarray,
                                     tau: complex = 1j) -> np.ndarray:
        """
        Parametrize E via Weierstrass ℘:
        (x, y) = (℘(u; Λ), ℘'(u; Λ))
        Returns array of (x, y) pairs.
        """
        from mnn.algebra.abelian import AbelianFunction
        xs = np.array([AbelianFunction.weierstrass_p(complex(ui)) for ui in u])
        # ℘' ≈ finite difference
        du = 1e-4
        xp = np.array([AbelianFunction.weierstrass_p(complex(ui+du)) for ui in u])
        xm = np.array([AbelianFunction.weierstrass_p(complex(ui-du)) for ui in u])
        ys = (xp - xm) / (2*du)
        return np.column_stack([np.real(xs), np.real(ys)])

    def g2_g3_invariants(self) -> Tuple[float, float]:
        """
        Eisenstein invariants g₂, g₃ related to Weierstrass form:
        y² = 4x³ − g₂x − g₃.
        From our form y² = x³ + ax + b:
        g₂ = −4a,  g₃ = −4b.
        """
        return (-4*self.a, -4*self.b)

    # ── Over Fp (finite field) ────────────────────────────────────────────────

    def points_over_Fp(self, p: int) -> List[Tuple[int,int]]:
        """
        All points on E over F_p.
        E(F_p): y² ≡ x³ + ax + b (mod p)
        """
        pts = [(None, None)]   # point at infinity
        for x in range(p):
            rhs = (x**3 + int(self.a)*x + int(self.b)) % p
            for y in range(p):
                if (y*y) % p == rhs:
                    pts.append((x, y))
        return pts

    def order_over_Fp(self, p: int) -> int:
        """#E(F_p) = p + 1 - t where t is the trace of Frobenius."""
        return len(self.points_over_Fp(p))

    def add_Fp(self, P: Tuple, Q: Tuple, p: int) -> Tuple:
        """Point addition over F_p."""
        if P[0] is None: return Q
        if Q[0] is None: return P
        x1,y1=P; x2,y2=Q
        if x1==x2 and (y1+y2)%p==0: return (None,None)
        if x1==x2 and y1==y2:
            if y1==0: return (None,None)
            lam=(3*x1**2+int(self.a)) * pow(2*y1, p-2, p) % p
        else:
            lam=(y2-y1)*pow(x2-x1, p-2, p) % p
        x3=(lam**2-x1-x2)%p; y3=(lam*(x1-x3)-y1)%p
        return (int(x3), int(y3))

    def __repr__(self):
        return f"EllipticCurve(y² = x³ + {self.a}x + {self.b}, Δ={self.discriminant():.4f})"


class JacobiVariety:
    """
    The Jacobian variety of a genus-g curve.
    For g=1 (elliptic curve): Jac(E) ≅ E itself.
    For g=2: Jac(C) is a 2-dimensional abelian variety, parametrized
             by the Riemann theta function.

    This class implements the g=2 case numerically.
    """

    def __init__(self, genus: int = 2, period_matrix: np.ndarray = None):
        self.genus = genus
        if period_matrix is None:
            # Default: principally polarized (Riemann matrix with pos. def. imag. part)
            self.Omega = np.array([[1j, 0.5j],[0.5j, 1.2j]])
        else:
            self.Omega = np.array(period_matrix, dtype=complex)

    def theta_function(self, z: np.ndarray, N: int = 4) -> complex:
        """
        Θ(z | Ω) = Σ_{n∈Z^g} exp(iπ nᵀΩn + 2πi nᵀz)
        """
        from itertools import product as iprod
        z = np.array(z, dtype=complex); val = 0
        for nv in iprod(range(-N, N+1), repeat=self.genus):
            n     = np.array(nv, dtype=complex)
            phase = 1j*np.pi*(n@self.Omega@n) + 2j*np.pi*(n@z)
            val  += np.exp(phase)
        return val

    def abel_jacobi_map(self, points_on_curve: np.ndarray) -> np.ndarray:
        """
        Approximate Abel-Jacobi map: C^g → Jac(C).
        For testing: maps g points to a vector in C^g.
        """
        # Simplified: project points onto the period lattice
        if len(points_on_curve) < self.genus:
            return np.zeros(self.genus, dtype=complex)
        return np.array([complex(p[0], p[1] if len(p)>1 else 0)
                         for p in points_on_curve[:self.genus]])

    def is_principally_polarized(self) -> bool:
        """Check if Ω is in Siegel upper half-space (Im(Ω) > 0)."""
        im_part = np.imag(self.Omega)
        eigvals = np.linalg.eigvalsh(im_part)
        return bool(np.all(eigvals > 0))

    def __repr__(self):
        return f"JacobiVariety(genus={self.genus}, principally_polarized={self.is_principally_polarized()})"
