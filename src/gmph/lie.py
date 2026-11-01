"""Матричные алгебры Ли: коммутатор, структурные константы, форма Киллинга.

Базис алгебры — список матриц SymPy T_1, …, T_n (линейно независимых).
Структурные константы определяются равенством [T_a, T_b] = Σ_c f_ab^c T_c;
присоединённое представление (ad_a)^c_b = f_ab^c, форма Киллинга
B_ab = tr(ad_a ad_b). Индексы в Python нумеруются с 0.
"""

import sympy as sp


def commutator(A, B):
    """[A, B] = AB − BA."""
    return A * B - B * A


def coordinates(X, basis):
    """Координаты матрицы X в базисе: X = Σ c_a T_a (ошибка, если X вне линейной оболочки)."""
    M = sp.Matrix.hstack(*[sp.Matrix(T).reshape(len(T), 1) for T in basis])
    target = sp.Matrix(X).reshape(len(X), 1)
    coeffs, params = M.gauss_jordan_solve(target)
    if params.shape[0]:
        raise ValueError("матрицы базиса линейно зависимы")
    return [sp.simplify(c) for c in coeffs]


def structure_constants(basis):
    """Массив f[a][b][c] с [T_a, T_b] = Σ_c f[a][b][c] T_c."""
    n = len(basis)
    return [[coordinates(commutator(basis[a], basis[b]), basis) for b in range(n)] for a in range(n)]


def adjoint(basis):
    """Матрицы присоединённого представления: (ad_a)[c, b] = f_ab^c."""
    f = structure_constants(basis)
    n = len(basis)
    return [sp.Matrix(n, n, lambda c, b: f[a][b][c]) for a in range(n)]


def killing_form(basis):
    """Форма Киллинга B_ab = tr(ad_a ad_b)."""
    ad = adjoint(basis)
    n = len(basis)
    return sp.Matrix(n, n, lambda a, b: (ad[a] * ad[b]).trace())


def spin_matrices(j):
    """Эрмитовы матрицы (Ĵ_x, Ĵ_y, Ĵ_z) неприводимого представления SU(2) со спином j (ħ = 1).

    Базис |j, m⟩ упорядочен по убыванию m = j, j − 1, …, −j; Ĵ_z = diag(m),
    Ĵ_+|m⟩ = √((j − m)(j + m + 1)) |m + 1⟩ (фазы Кондона–Шортли).
    Связь с вещественными генераторами конвенций: Ĵ_k = i ρ(J_k), [Ĵ_x, Ĵ_y] = i Ĵ_z.
    """
    j = sp.nsimplify(j)
    n = int(2 * j + 1)
    ms = [j - a for a in range(n)]
    Jz = sp.diag(*ms)
    Jp = sp.zeros(n, n)
    for a in range(1, n):
        m = ms[a]
        Jp[a - 1, a] = sp.sqrt((j - m) * (j + m + 1))
    Jm = Jp.T
    return (Jp + Jm) / 2, (Jp - Jm) / (2 * sp.I), Jz
