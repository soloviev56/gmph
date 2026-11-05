import numpy as np
import pytest

from gmph.dec import betti_numbers, edge_endpoints, incidence, node_coords


@pytest.mark.parametrize("periodic", [False, True])
def test_d_squared_is_zero(periodic):
    D0, D1 = incidence(4, 3, periodic)
    assert abs(D1 @ D0).sum() == 0


def test_discrete_stokes_and_gradient():
    nx, ny, h = 5, 4, 0.3
    D0, D1 = incidence(nx, ny)
    xy = node_coords(nx, ny, h, h)
    f = np.sin(xy[:, 0]) * xy[:, 1] ** 2
    start, end = edge_endpoints(nx, ny, h, h)
    g = lambda p: np.sin(p[:, 0]) * p[:, 1] ** 2
    assert np.allclose(D0 @ f, g(end) - g(start))
    # 1-форма x dy: интеграл по ребру; d(x dy) = dx∧dy, интеграл по ячейке h²
    mid = (start + end) / 2
    a = np.where(start[:, 0] == end[:, 0], mid[:, 0] * h, 0.0)
    assert np.allclose(D1 @ a, h * h)


def test_betti_numbers():
    assert betti_numbers(4, 3) == (1, 0, 0)            # квадрат стягиваем
    assert betti_numbers(4, 3, periodic=True) == (1, 2, 1)  # тор
