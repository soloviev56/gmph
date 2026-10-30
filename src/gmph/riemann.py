"""Связности в координатах (раздел «Риманова геометрия»).

Соглашение (страница «Конвенции»): ∇_{∂_j} ∂_k = Γ^i_{jk} ∂_i — первый нижний
индекс указывает направление дифференцирования. Символы хранятся как
``Gamma[i][j][k]`` (вложенные списки выражений SymPy).
"""

import itertools

import sympy as sp


def flat_connection(x_of_y, y):
    """Символы евклидовой (плоской) связности в координатах y.

    ``x_of_y`` — декартовы координаты как функции y. В декартовых Γ = 0,
    в координатах y: Γ^i_{jk} = (∂y^i/∂x^a) ∂²x^a/∂y^j∂y^k.
    """
    n = len(y)
    J = sp.Matrix(n, n, lambda a, j: sp.diff(x_of_y[a], y[j]))
    Jinv = sp.simplify(J.inv())
    return [[[sp.simplify(sum(Jinv[i, a] * sp.diff(x_of_y[a], y[j], y[k]) for a in range(n)))
              for k in range(n)] for j in range(n)] for i in range(n)]


def covariant_derivative(T, kinds, Gamma, coords):
    """Ковариантная производная тензора с компонентами ``T`` (sympy Array).

    ``kinds`` — строка из 'u' (верхний индекс) и 'd' (нижний) для каждого
    индекса T. Результат R[j, ...] = ∇_j T^{...}_{...}: индекс производной первый.
    """
    T = sp.Array(T)
    n = len(coords)
    rank = len(kinds)
    shape = (n,) * (rank + 1)
    result = sp.MutableDenseNDimArray.zeros(*shape)
    for j in range(n):
        for idx in itertools.product(range(n), repeat=rank):
            value = sp.diff(T[idx], coords[j])
            for pos, kind in enumerate(kinds):
                for m in range(n):
                    moved = idx[:pos] + (m,) + idx[pos + 1:]
                    if kind == "u":
                        value += Gamma[idx[pos]][j][m] * T[moved]
                    else:
                        value -= Gamma[m][j][idx[pos]] * T[moved]
            result[(j,) + idx] = sp.simplify(value)
    return sp.Array(result)


def torsion(Gamma):
    """Тензор кручения T^i_{jk} = Γ^i_{jk} − Γ^i_{kj}."""
    n = len(Gamma)
    return [[[sp.simplify(Gamma[i][j][k] - Gamma[i][k][j]) for k in range(n)] for j in range(n)] for i in range(n)]
