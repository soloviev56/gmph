"""Калибровочные поля в физической записи конвенций (раздел «Калибровочные поля»).

Связность — список эрмитовых матриц 𝒜_μ (компоненты 1-формы), D_μ = ∂_μ + i𝒜_μ:

    ψ → Uψ,   𝒜_μ → U𝒜_μU⁻¹ + i(∂_μU)U⁻¹,   ℱ_μν = ∂_μ𝒜_ν − ∂_ν𝒜_μ + i[𝒜_μ, 𝒜_ν].

Параллельный перенос D_tψ = 0 вдоль x(t): dψ/dt = −i𝒜_μ(x)ẋ^μ ψ, ψ(t₁) = W ψ(t₀);
при калибровочном преобразовании W → U(x(t₁)) W U(x(t₀))⁻¹.
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

__all__ = ["gauge_transform", "field_strength", "adjoint_derivative", "wilson_line"]


def _as_matrix(a):
    return a if isinstance(a, sp.MatrixBase) else sp.Matrix([[a]])


def gauge_transform(A, U, coords, simplify=True):
    """Компоненты 𝒜'_μ = U𝒜_μU⁻¹ + i(∂_μU)U⁻¹ (U — sympy-матрица или число)."""
    U = _as_matrix(U)
    U_inv = U.inv()
    out = [U * _as_matrix(a) * U_inv + sp.I * U.diff(x) * U_inv for a, x in zip(A, coords)]
    return [sp.simplify(m) for m in out] if simplify else out


def field_strength(A, coords, simplify=True):
    """Словарь {(μ, ν): ℱ_μν} при μ < ν, ℱ_μν = ∂_μ𝒜_ν − ∂_ν𝒜_μ + i[𝒜_μ, 𝒜_ν]."""
    A = [_as_matrix(a) for a in A]
    n = len(coords)
    F = {}
    for m in range(n):
        for k in range(m + 1, n):
            f = A[k].diff(coords[m]) - A[m].diff(coords[k]) + sp.I * (A[m] * A[k] - A[k] * A[m])
            F[(m, k)] = sp.simplify(f) if simplify else f
    return F


def adjoint_derivative(A, eps, coords):
    """D_με = ∂_με + i[𝒜_μ, ε] — ковариантная производная в присоединённом представлении."""
    eps = _as_matrix(eps)
    return [eps.diff(x) + sp.I * (_as_matrix(a) * eps - eps * _as_matrix(a)) for a, x in zip(A, coords)]


def wilson_line(A_num, path, velocity, t0, t1, rtol=1e-10, atol=1e-12):
    """Перенос W: dW/dt = −i 𝒜_μ(x(t)) ẋ^μ W, W(t₀) = 1.

    A_num(x) — список числовых матриц 𝒜_μ в точке x; path(t), velocity(t) — x(t) и ẋ(t).
    """
    n = np.atleast_2d(A_num(path(t0))[0]).shape[0]

    def rhs(t, w):
        a = sum(np.atleast_2d(a_mu) * v for a_mu, v in zip(A_num(path(t)), velocity(t)))
        return (-1j * a @ w.reshape(n, n)).ravel()

    sol = solve_ivp(rhs, (t0, t1), np.eye(n, dtype=complex).ravel(), method="DOP853", rtol=rtol, atol=atol)
    return sol.y[:, -1].reshape(n, n)
