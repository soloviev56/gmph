"""Гамильтоновы поля и скобки Пуассона для симплектической формы общего вида.

Симплектическая форма — словарь из ``gmph.exterior`` в координатах
``coords``. Соглашения — со страницы «Конвенции»: ι_{X_H} ω = −dH,
{f, g} = X_g(f) = ω(X_g, X_f). Для ω = Σ dp_i∧dq^i при порядке координат
(q¹…qⁿ, p₁…pₙ) матрица формы — Ω = [[0, −1], [1, 0]], а X_H = Ω⁻¹∇H.
"""

import sympy as sp


def form_matrix(form, n):
    """Матрица ω_ij = ω(∂_i, ∂_j) 2-формы в n-мерной карте."""
    W = sp.zeros(n, n)
    for (i, j), c in form.items():
        W[i - 1, j - 1] += c
        W[j - 1, i - 1] -= c
    return W


def hamiltonian_field(H, coords, omega):
    """Компоненты поля X_H из уравнения ι_X ω = −dH, то есть X = W⁻¹∇H."""
    W = form_matrix(omega, len(coords))
    grad = sp.Matrix([sp.diff(H, x) for x in coords])
    return [sp.simplify(c) for c in W.inv() * grad]


def poisson(f, g, coords, omega):
    """Скобка Пуассона {f, g} = X_g(f)."""
    Xg = hamiltonian_field(g, coords, omega)
    return sp.simplify(sum(c * sp.diff(f, x) for c, x in zip(Xg, coords)))


def canonical_omega(n):
    """ω = Σ dp_i∧dq^i в координатах (q¹…qⁿ, p₁…pₙ)."""
    return {(i, n + i): -1 for i in range(1, n + 1)}
