import numpy as np
import pytest
import scipy.linalg as sla

from gmph.fem import (barycentric_gradients, edges, incidence, p1_matrices, square_mesh,
                      whitney1_at_centroids, whitney1_matrices)


@pytest.mark.parametrize("pattern", ["right", "crisscross"])
def test_mesh_orientation_and_euler(pattern):
    points, tri = square_mesh(3, 2.0, pattern)
    area, _ = barycentric_gradients(points, tri)
    assert np.all(area > 0) and np.isclose(area.sum(), 4.0)
    edge, _, _ = edges(tri)
    assert len(points) - len(edge) + len(tri) == 1            # χ(диска) = 1


@pytest.mark.parametrize("pattern", ["right", "crisscross"])
def test_d_squared_and_gradients(pattern):
    points, tri = square_mesh(4, 1.0, pattern)
    D0, D1 = incidence(points, tri)
    assert abs(D1 @ D0).sum() == 0
    # градиент P1-функции = форма Уитни с коэффициентами D0 f (в центрах треугольников)
    f = np.sin(points[:, 0]) + points[:, 1] ** 2
    _, g = barycentric_gradients(points, tri)
    grad = np.einsum("tk,tka->ta", f[tri], g)
    assert np.allclose(whitney1_at_centroids(points, tri, D0 @ f), grad)


def test_p1_matrices():
    points, tri = square_mesh(4, 1.0)
    K, M = p1_matrices(points, tri)
    one = np.ones(len(points))
    assert np.allclose(K @ one, 0) and np.isclose(one @ M @ one, 1.0)
    x = points[:, 0]
    assert np.isclose(x @ K @ x, 1.0)                         # ∫|∇x|² = 1


def test_whitney1_kernel_and_spectrum():
    n = 8
    points, tri = square_mesh(n, np.pi)
    C, M = whitney1_matrices(points, tri)
    D0, _ = incidence(points, tri)
    assert abs(C @ D0).max() < 1e-12                         # rot grad = 0
    lam = sla.eigh(C.toarray(), M.toarray(), eigvals_only=True)
    zeros = np.sum(lam < 1e-8)
    assert zeros == len(points) - 1                           # ядро — градиенты (без констант)
    # свободная граница: rot u = 0 на краю, спектр задачи Дирихле m² + n², m, n ≥ 1
    assert np.allclose(lam[zeros:zeros + 3], [2, 5, 5], rtol=0.05)


@pytest.mark.parametrize("order", [1, 2])
def test_elasticity_rigid_modes_and_patch(order):
    from gmph.fem import elasticity_system
    points, tri = square_mesh(3, 1.0)
    sysm = elasticity_system(points, tri, order, lam=2.0, mu=1.0)
    K, nodes = sysm["K"].toarray(), sysm["nodes"]
    x, y = nodes[:, 0], nodes[:, 1]
    rigid = [np.ravel(np.column_stack(c)) for c in ((1 + 0 * x, 0 * x), (0 * x, 1 + 0 * x), (-y, x))]
    for r in rigid:
        assert np.abs(K @ r).max() < 1e-12
    assert np.sum(np.linalg.eigvalsh(K) < 1e-10) == 3
    # однородная деформация u = (x, 0): энергия ∫σ:ε = λ + 2μ, ∫div u = площадь
    u = np.ravel(np.column_stack((x, 0 * x)))
    assert np.isclose(u @ K @ u, 2.0 + 2.0)
    assert np.isclose(sysm["B"] @ u @ np.ones(len(tri)), 1.0)


def test_p2_basis_partition_and_quadrature():
    from gmph.fem import lagrange_basis, triangle_quadrature
    b, w = triangle_quadrature(4)
    assert np.allclose(lagrange_basis(2, b).sum(axis=1), 1) and np.isclose(w.sum(), 1)
    # ∫λ0²λ1² по треугольнику площади 1 = 2!2!2!/6! · 2 = 1/90
    assert np.isclose(np.sum(w * b[:, 0] ** 2 * b[:, 1] ** 2), 1 / 90)


def test_rectangle_mesh():
    from gmph.fem import rectangle_mesh
    points, tri = rectangle_mesh(4, 2, 4.0, 1.0)
    area, _ = barycentric_gradients(points, tri)
    assert np.all(area > 0) and np.isclose(area.sum(), 4.0) and len(tri) == 16
