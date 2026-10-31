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


def induced_connection(X, coords):
    """Символы связности поверхности, вложенной в евклидово пространство.

    ``X`` — декартовы координаты точки поверхности как функции ``coords``.
    Производная поля вдоль поверхности проецируется на касательную плоскость:
    Γ^i_{jk} = g^{il} (∂_l X · ∂_j ∂_k X), g_{jl} = ∂_j X · ∂_l X.
    """
    n = len(coords)
    E = [[sp.diff(c, q) for c in X] for q in coords]          # ∂_j X
    dot = lambda u, v: sum(a * b for a, b in zip(u, v))
    g = sp.Matrix(n, n, lambda j, l: sp.simplify(dot(E[j], E[l])))
    ginv = sp.simplify(g.inv())
    second = [[[sp.diff(c, coords[j], coords[k]) for c in X] for k in range(n)] for j in range(n)]
    return [[[sp.simplify(sum(ginv[i, l] * dot(E[l], second[j][k]) for l in range(n)))
              for k in range(n)] for j in range(n)] for i in range(n)]


def christoffel(g, coords):
    """Символы Кристоффеля Γ^i_{jk} = ½ g^{il} (∂_j g_{lk} + ∂_k g_{lj} − ∂_l g_{jk})."""
    g = sp.Matrix(g)
    n = len(coords)
    ginv = sp.simplify(g.inv())
    return [[[sp.simplify(sum(ginv[i, l] * (sp.diff(g[l, k], coords[j]) + sp.diff(g[l, j], coords[k])
                                            - sp.diff(g[j, k], coords[l])) for l in range(n)) / 2)
              for k in range(n)] for j in range(n)] for i in range(n)]


def numeric_connection(Gamma, coords):
    """Функция x ↦ массив numpy Γ[i, j, k] для численного интегрирования."""
    import numpy as np

    f = sp.lambdify(coords, Gamma, "numpy")
    n = len(coords)
    return lambda x: np.array(f(*x), dtype=float).reshape(n, n, n)


def geodesic_rhs(Gamma, coords):
    """Правая часть уравнения геодезических для ``solve_ivp``: состояние (x, ẋ).

    ẍ^i = −Γ^i_{jk} ẋ^j ẋ^k.
    """
    import numpy as np

    G = numeric_connection(Gamma, coords)
    n = len(coords)

    def rhs(t, state):
        x, v = state[:n], state[n:]
        return np.concatenate([v, -np.einsum("ijk,j,k->i", G(x), v, v)])
    return rhs


def transport_rhs(Gamma, coords, path, velocity):
    """Правая часть уравнения параллельного переноса dV^i/dt = −Γ^i_{jk} ẋ^j V^k.

    ``path(t)`` и ``velocity(t)`` — кривая и её скорость (массивы numpy).
    """
    import numpy as np

    G = numeric_connection(Gamma, coords)
    return lambda t, V: -np.einsum("ijk,j,k->i", G(path(t)), velocity(t), V)
