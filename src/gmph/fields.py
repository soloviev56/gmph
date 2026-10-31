"""Лагранжева теория поля: уравнения Эйлера–Лагранжа и токи Нётер.

Поля задаются функциями SymPy от координат ``coords``, например
``phi = sp.Function("phi")(t, x)``; плотность лагранжиана ℒ — выражение
от полей и их первых производных.
"""

import sympy as sp


def euler_lagrange(L, fields, coords):
    """Левые части уравнений ∂_μ (∂ℒ/∂(∂_μφ)) − ∂ℒ/∂φ = 0, по одной на поле."""
    return [sp.expand(sum(sp.diff(sp.diff(L, sp.diff(f, x)), x) for x in coords) - sp.diff(L, f))
            for f in fields]


def momentum_densities(L, fields, time):
    """Плотности канонических импульсов π = ∂ℒ/∂(∂_t φ)."""
    return [sp.diff(L, sp.diff(f, time)) for f in fields]


def canonical_stress(L, fields, coords):
    """Канонический тензор T^μ_ν = Σ ∂ℒ/∂(∂_μφ) ∂_νφ − δ^μ_ν ℒ (массив n×n)."""
    n = len(coords)
    return sp.Matrix(n, n, lambda m, v: sum(sp.diff(L, sp.diff(f, coords[m])) * sp.diff(f, coords[v]) for f in fields)
                     - (L if m == v else 0))


def divergence(T, coords):
    """∂_μ T^μ_ν — по одному выражению на значение ν."""
    n = len(coords)
    return [sp.simplify(sum(sp.diff(T[m, v], coords[m]) for m in range(n))) for v in range(n)]
