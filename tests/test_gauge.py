import numpy as np
import sympy as sp

from gmph.gauge import adjoint_derivative, field_strength, gauge_transform, wilson_line

x, y = sp.symbols("x y", real=True)
sigma = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.Matrix([[1, 0], [0, -1]])]


def su2(a, n):
    """exp(i a n·σ) для единичного n."""
    return sp.cos(a) * sp.eye(2) + sp.I * sp.sin(a) * sum((ni * s for ni, s in zip(n, sigma)), sp.zeros(2))


def test_abelian_gauge():
    A = [sp.Function("ax")(x, y), sp.Function("ay")(x, y)]
    al = sp.Function("alpha")(x, y)
    A2 = gauge_transform(A, sp.exp(sp.I * al), (x, y))
    assert sp.simplify(A2[0][0] - (A[0] - al.diff(x))) == 0
    F, F2 = field_strength(A, (x, y)), field_strength(A2, (x, y))
    assert sp.simplify(F2[(0, 1)] - F[(0, 1)]) == sp.zeros(1)


def poly_su2_connection():
    """Конкретная (не плоская) su(2)-связность с полиномиальными коэффициентами."""
    return [(x * y * sigma[0] + (1 - x) * sigma[2]) / 2, (y**2 * sigma[1] + x * sigma[0] - y * sigma[2]) / 3]


def test_curvature_covariant():
    A = poly_su2_connection()
    U = su2(x * y + x, (sp.Rational(3, 5), 0, sp.Rational(4, 5)))
    F = field_strength(A, (x, y))[(0, 1)]
    F2 = field_strength(gauge_transform(A, U, (x, y), simplify=False), (x, y), simplify=False)[(0, 1)]
    for pt in ({x: sp.Rational(1, 3), y: sp.Rational(-2, 7)}, {x: 2, y: sp.Rational(1, 5)}):
        diff = (F2 - U * F * U.inv()).subs(pt)
        assert max(abs(sp.N(v, 30)) for v in diff) < 1e-25


def test_infinitesimal_gauge():
    A = poly_su2_connection()
    e = sp.Symbol("e")
    eps = x * sigma[2] + y**2 * sigma[0] - x * y * sigma[1]
    A2 = gauge_transform(A, sp.eye(2) + sp.I * e * eps, (x, y), simplify=False)
    for m, D in enumerate(adjoint_derivative(A, eps, (x, y))):
        first = (A2[m] - A[m]).applyfunc(lambda z: sp.diff(z, e).subs(e, 0))
        assert sp.expand(first + D) == sp.zeros(2)          # δ𝒜 = −Dε


def test_wilson_line_covariance():
    A = [sp.Matrix([[y, x - sp.I * y], [x + sp.I * y, -y]]) / 2, sp.Matrix([[x * y, 1], [1, -x * y]])]
    U = su2(x**2 - y, (0, sp.Rational(3, 5), sp.Rational(4, 5)))
    A2 = gauge_transform(A, U, (x, y), simplify=False)
    num = lambda comps: sp.lambdify((x, y), comps, "numpy")
    f1, f2, fU = num(A), num(A2), sp.lambdify((x, y), U, "numpy")
    path = lambda t: (np.cos(t), 0.5 * np.sin(t))
    vel = lambda t: (-np.sin(t), 0.5 * np.cos(t))
    W1 = wilson_line(lambda p: f1(*p), path, vel, 0.0, 2.0)
    W2 = wilson_line(lambda p: f2(*p), path, vel, 0.0, 2.0)
    U1, U0 = np.array(fU(*path(2.0)), complex), np.array(fU(*path(0.0)), complex)
    assert np.allclose(W2, U1 @ W1 @ np.linalg.inv(U0), atol=1e-8)
    assert np.allclose(W1.conj().T @ W1, np.eye(2), atol=1e-9)    # эрмитова 𝒜 → унитарный перенос
