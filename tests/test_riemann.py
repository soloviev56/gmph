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
