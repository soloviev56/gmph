import sympy as sp

from gmph.cartan import (connection_forms, curvature_forms, frame_metric, frame_ricci, frame_riemann,
                         frame_scalar)
from gmph.riemann import christoffel, riemann


def coordinate_to_frame(R, E):
    """R^a_bcd = E^a_i R^i_jkl e_b^j e_c^k e_d^l."""
    E = sp.Matrix(E)
    V = E.inv()
    n = E.shape[0]
    r = range(n)
    return [[[[sp.simplify(sum(E[a, i] * R[i][j][k][l] * V[j, b] * V[k, c] * V[l, dd]
                               for i in r for j in r for k in r for l in r))
               for dd in r] for c in r] for b in r] for a in r]


def test_sphere():
    th, ph, a = sp.symbols("theta phi a", positive=True)
    E = sp.diag(a, a * sp.sin(th))
    eta = sp.eye(2)
    om = connection_forms(E, (th, ph), eta)
    assert sp.simplify(om[0][1][(2,)] + sp.cos(th)) == 0          # ω^1_2 = −cos θ dφ
    Om = curvature_forms(om, (th, ph))
    assert sp.simplify(Om[0][1][(1, 2)] - sp.sin(th)) == 0         # Ω^1_2 = sin θ dθ∧dφ
    R = frame_riemann(Om, E)
    assert sp.simplify(R[0][1][0][1] - 1 / a**2) == 0
    assert sp.simplify(frame_scalar(frame_ricci(R), eta) - 2 / a**2) == 0


def test_agrees_with_christoffel_route():
    t, r, th, ph = sp.symbols("t r theta phi", positive=True)
    f = sp.Function("f", positive=True)(r)
    h = sp.Function("h", positive=True)(r)
    coords = (t, r, th, ph)
    E = sp.diag(f, h, r, r * sp.sin(th))                     # общий статический сферически симметричный
    eta = sp.diag(1, -1, -1, -1)
    g = frame_metric(E, eta)
    R_coord = riemann(christoffel(g, coords), coords)
    R_frame = frame_riemann(curvature_forms(connection_forms(E, coords, eta), coords), E)
    expected = coordinate_to_frame(R_coord, E)
    for idx in [(0, 1, 0, 1), (0, 2, 0, 2), (1, 2, 1, 2), (2, 3, 2, 3), (1, 3, 1, 3), (0, 3, 0, 3),
                (1, 0, 1, 0), (0, 1, 2, 3), (2, 1, 2, 1)]:
        a, b, c, dd = idx
        assert sp.simplify(R_frame[a][b][c][dd] - expected[a][b][c][dd]) == 0
