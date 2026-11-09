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
