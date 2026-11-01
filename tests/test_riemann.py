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


theta = sp.Symbol("theta", positive=True)
SPHERE = [sp.sin(theta) * sp.cos(phi), sp.sin(theta) * sp.sin(phi), sp.cos(theta)]


def test_sphere_connection_is_christoffel():
    from gmph.riemann import christoffel, induced_connection

    G = induced_connection(SPHERE, (theta, phi))
    assert sp.simplify(G[0][1][1] + sp.sin(theta) * sp.cos(theta)) == 0      # Γ^θ_{φφ}
    assert sp.simplify(G[1][0][1] - sp.cos(theta) / sp.sin(theta)) == 0      # Γ^φ_{θφ}
    C = christoffel([[1, 0], [0, sp.sin(theta)**2]], (theta, phi))
    assert all(sp.simplify(G[i][j][k] - C[i][j][k]) == 0 for i in range(2) for j in range(2) for k in range(2))
    assert christoffel([[1, 0], [0, r**2]], (r, phi)) == flat_connection(POLAR, (r, phi))


def test_latitude_holonomy():
    import numpy as np
    from scipy.integrate import solve_ivp

    from gmph.riemann import induced_connection, transport_rhs

    th0 = 1.0
    G = induced_connection(SPHERE, (theta, phi))
    rhs = transport_rhs(G, (theta, phi), lambda t: np.array([th0, t]), lambda t: np.array([0.0, 1.0]))
    V = solve_ivp(rhs, (0, 2 * np.pi), [1.0, 0.0], rtol=1e-11, atol=1e-12).y[:, -1]
    z = V[0] + 1j * np.sin(th0) * V[1]                  # компоненты в ортах e_θ, e_φ
    assert abs(z - np.exp(-2j * np.pi * np.cos(th0))) < 1e-8   # поворот на −2π cos θ₀


def test_sphere_curvature():
    from gmph.riemann import christoffel, ricci, riemann, scalar_curvature

    a = sp.Symbol("a", positive=True)
    g = sp.diag(a**2, a**2 * sp.sin(theta)**2)
    R = riemann(christoffel(g, (theta, phi)), (theta, phi))
    assert sp.simplify(R[0][1][0][1] - sp.sin(theta)**2) == 0          # R^θ_{φθφ}
    Ric = ricci(R)
    assert sp.simplify(Ric - g / a**2) == sp.zeros(2, 2)
    assert sp.simplify(scalar_curvature(Ric, g) - 2 / a**2) == 0


def test_flat_curvature_vanishes():
    from gmph.riemann import riemann

    R = riemann(flat_connection(POLAR, (r, phi)), (r, phi))
    assert all(c == 0 for a in R for b in a for e in b for c in e)


def test_commutator_of_covariant_derivatives():
    from gmph.riemann import christoffel, riemann

    g = sp.diag(1, sp.sin(theta)**2)
    q = (theta, phi)
    G = christoffel(g, q)
    R = riemann(G, q)
    V = [sp.Function("V0")(*q), sp.Function("V1")(*q)]
    DDV = covariant_derivative(covariant_derivative(V, "u", G, q), "du", G, q)   # [k, l, i] = ∇_k∇_l V^i
    for k in range(2):
        for l in range(2):
            for i in range(2):
                rhs = sum(R[i][j][k][l] * V[j] for j in range(2))
                assert sp.simplify(DDV[k, l, i] - DDV[l, k, i] - rhs) == 0
