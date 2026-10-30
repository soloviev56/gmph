import sympy as sp

from gmph.riemann import covariant_derivative, flat_connection, torsion

r, phi = sp.symbols("r phi", positive=True)
POLAR = [r * sp.cos(phi), r * sp.sin(phi)]


def test_polar_symbols():
    G = flat_connection(POLAR, (r, phi))
    assert G[0][1][1] == -r                       # Γ^r_{φφ}
    assert G[1][0][1] == 1 / r and G[1][1][0] == 1 / r
    assert G[0][0][0] == 0 and G[1][1][1] == 0
    assert all(c == 0 for row in torsion(G) for col in row for c in col)


def test_constant_field_is_parallel():
    G = flat_connection(POLAR, (r, phi))
    d_x = [sp.cos(phi), -sp.sin(phi) / r]          # ∂_x в полярных компонентах
    assert covariant_derivative(d_x, "u", G, (r, phi)) == sp.Array([[0, 0], [0, 0]])
    metric = [[1, 0], [0, r**2]]
    assert covariant_derivative(metric, "dd", G, (r, phi)) == sp.Array.zeros(2, 2, 2)
