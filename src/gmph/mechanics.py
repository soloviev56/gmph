"""Лагранжева механика: уравнения движения по функции Лагранжа.

Обобщённые координаты задаются как функции времени SymPy, например
``phi = sp.Function("phi")(t)``; лагранжиан — выражение от них и их
производных по ``t``. Функции возвращают обычные выражения SymPy или, для
численного интегрирования, функции состояния ``y = (q¹, …, qⁿ, q̇¹, …, q̇ⁿ)``.
"""

import numpy as np
import sympy as sp


def euler_lagrange(L, coords, t):
    """Левые части уравнений Лагранжа d/dt ∂L/∂q̇ⁱ − ∂L/∂qⁱ, по одной на координату."""
    return [sp.diff(L.diff(q.diff(t)), t) - L.diff(q) for q in coords]


def momenta(L, coords, t):
    """Обобщённые импульсы pᵢ = ∂L/∂q̇ⁱ."""
    return [L.diff(q.diff(t)) for q in coords]


def energy(L, coords, t):
    """Функция энергии E = q̇ⁱ ∂L/∂q̇ⁱ − L."""
    return sum(q.diff(t) * p for q, p in zip(coords, momenta(L, coords, t))) - L


def mass_matrix(L, coords, t):
    """Матрица кинетической энергии Mᵢⱼ = ∂²L/∂q̇ⁱ∂q̇ʲ."""
    v = [q.diff(t) for q in coords]
    return sp.Matrix(len(v), len(v), lambda i, j: sp.diff(L, v[i], v[j]))


def accelerations(L, coords, t, simplify=True):
    """Решить уравнения Лагранжа относительно ускорений: список q̈ⁱ(q, q̇, t).

    Уравнения линейны по ускорениям с матрицей ∂²L/∂q̇∂q̇, поэтому решаются
    как линейная система (матрица должна быть невырожденной).
    """
    acc = [q.diff(t, 2) for q in coords]
    a = sp.symbols(f"a0:{len(coords)}")
    eqs = [e.subs(dict(zip(acc, a))) for e in euler_lagrange(L, coords, t)]
    M, rhs = sp.linear_eq_to_matrix(eqs, a)
    solution = list(M.LUsolve(rhs))
    return [sp.simplify(s) for s in solution] if simplify else solution


def state_function(expr, coords, t, params=None):
    """Числовая функция f(t, y) от выражения в q(t), q̇(t).

    ``y = (q¹, …, qⁿ, q̇¹, …, q̇ⁿ)``; ``params`` — словарь значений параметров.
    ``expr`` может быть числом, выражением или списком выражений.
    """
    n = len(coords)
    qs = sp.symbols(f"q0:{n}")
    vs = sp.symbols(f"v0:{n}")
    subs_v = dict(zip([q.diff(t) for q in coords], vs))
    subs_q = dict(zip(coords, qs))

    def to_state(e):
        return sp.sympify(e).subs(params or {}).subs(subs_v).subs(subs_q)

    if isinstance(expr, (list, tuple)):
        state_expr = [to_state(e) for e in expr]
    else:
        state_expr = to_state(expr)
    numeric = sp.lambdify((t, *qs, *vs), state_expr, "numpy")
    return lambda time, y: numeric(time, *y)


def first_order_system(L, coords, t, params=None):
    """Правая часть системы первого порядка ẏ = f(t, y) для ``solve_ivp``."""
    n = len(coords)
    acc = state_function(accelerations(L, coords, t), coords, t, params)

    def rhs(time, y):
        return np.concatenate([np.asarray(y[n:], dtype=float), np.asarray(acc(time, y), dtype=float)])

    return rhs
