"""Метод подвижного репера Картана (статья «Тетрады и структурные уравнения Картана»).

Соглашения (страница «Конвенции», раздел «Связности»):
- корепер e^a = E[a, μ] dx^μ, метрика g = η_ab e^a e^b (η — диагональная матрица из ±1);
- форма связности: ∇e_b = e_a ω^a_b, ω_ab = η_ac ω^c_b антисимметрична;
- первое структурное уравнение (без кручения): de^a + ω^a_b ∧ e^b = 0;
- второе: Ω^a_b = dω^a_b + ω^a_c ∧ ω^c_b = ½ R^a_bcd e^c ∧ e^d, где R^a_bcd — компоненты
  тензора Римана R^i_jkl (соглашение gmph.riemann) в репере.

Формы хранятся как в ``gmph.exterior``: словарь {кортеж индексов (с 1): коэффициент}.
"""

import itertools

import sympy as sp

from gmph.exterior import add, d, scale, wedge
from gmph.riemann import _tidy


def coframe_forms(E):
    """Строки матрицы E как 1-формы e^a."""
    E = sp.Matrix(E)
    return [{(mu + 1,): E[a, mu] for mu in range(E.shape[1]) if E[a, mu] != 0} for a in range(E.shape[0])]


def frame_metric(E, eta):
    """Метрика g_μν = η_ab E^a_μ E^b_ν."""
    E = sp.Matrix(E)
    return sp.simplify(E.T * sp.Matrix(eta) * E)


def connection_forms(E, coords, eta):
    """Формы связности ω^a_b (матрица словарей) из первого структурного уравнения.

    Неизвестные — координатные компоненты ω_ab,μ при a < b; система
    de^a + ω^a_b ∧ e^b = 0 линейна и имеет единственное решение.
    """
    n = len(coords)
    eta = sp.Matrix(eta)
    e = coframe_forms(E)
    unknown = {(a, b): [sp.Dummy(f"w{a}{b}_{mu}") for mu in range(n)]
               for a, b in itertools.combinations(range(n), 2)}

    def lowered(a, b):                       # ω_ab как 1-форма с неизвестными коэффициентами
        if a == b:
            return {}
        if a < b:
            return {(mu + 1,): w for mu, w in enumerate(unknown[a, b])}
        return {(mu + 1,): -w for mu, w in enumerate(unknown[b, a])}

    def omega_up(a, b):                      # ω^a_b = η^aa ω_ab (η диагональна)
        return scale(1 / eta[a, a], lowered(a, b))

    equations = []
    for a in range(n):
        torsion = add(d(e[a], coords), *[wedge(omega_up(a, b), e[b]) for b in range(n)])
        equations += list(torsion.values())
    symbols = [w for ws in unknown.values() for w in ws]
    solution = sp.solve(equations, symbols, dict=True)[0]
    return [[{k: _tidy(v.subs(solution)) for k, v in omega_up(a, b).items() if _tidy(v.subs(solution)) != 0}
             for b in range(n)] for a in range(n)]


def curvature_forms(omega, coords):
    """Формы кривизны Ω^a_b = dω^a_b + ω^a_c ∧ ω^c_b."""
    n = len(omega)
    result = []
    for a in range(n):
        row = []
        for b in range(n):
            form = add(d(omega[a][b], coords), *[wedge(omega[a][c], omega[c][b]) for c in range(n)])
            row.append({k: _tidy(v) for k, v in form.items() if _tidy(v) != 0})
        result.append(row)
    return result


def frame_riemann(Omega, E):
    """Компоненты R^a_bcd = Ω^a_b(e_c, e_d) в репере (массив ``R[a][b][c][d]``)."""
    E = sp.Matrix(E)
    n = E.shape[0]
    vectors = sp.simplify(E.inv())           # столбец c — вектор e_c в координатном базисе

    def value(form, c, dd):
        return sum(coef * (vectors[i - 1, c] * vectors[j - 1, dd] - vectors[j - 1, c] * vectors[i - 1, dd])
                   for (i, j), coef in form.items())
    return [[[[_tidy(value(Omega[a][b], c, dd)) for dd in range(n)] for c in range(n)]
             for b in range(n)] for a in range(n)]


def frame_ricci(R):
    """Тензор Риччи в репере R_bd = R^a_bad (sympy Matrix)."""
    n = len(R)
    return sp.Matrix(n, n, lambda b, dd: _tidy(sum(R[a][b][a][dd] for a in range(n))))


def frame_scalar(Ric, eta):
    """Скалярная кривизна R = η^bd R_bd."""
    eta = sp.Matrix(eta)
    return sp.simplify(sum(Ric[b, b] / eta[b, b] for b in range(eta.shape[0])))
