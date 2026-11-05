"""Дискретное внешнее исчисление на прямоугольной сетке (статья «Дискретное внешнее исчисление и схема Йи»).

Сетка nx × ny ячеек. Коцепи — числа на клетках:
0-коцепь — на узлах, 1-коцепь — на рёбрах, 2-коцепь — на ячейках.

Нумерация (без периодичности):
- узел (i, j), 0 ≤ i ≤ nx, 0 ≤ j ≤ ny, имеет номер ``i + (nx + 1) j``;
- сначала идут x-рёбра (i, j) → (i + 1, j), номер ``i + nx j``,
  затем y-рёбра (i, j) → (i, j + 1), номер ``nx (ny + 1) + i + (nx + 1) j``;
- ячейка [i, i + 1] × [j, j + 1] имеет номер ``i + nx j`` и ориентирована
  против часовой стрелки.
При ``periodic=True`` индексы берутся по модулю nx, ny (сетка на торе), узлов и
ячеек по nx·ny, рёбер 2·nx·ny.

Дискретный дифференциал d — матрица инцидентности (кограничный оператор):
``(D0 f)[ребро] = f(конец) − f(начало)``, ``(D1 a)[ячейка]`` — сумма значений на рёбрах
её границы с учётом ориентации. Тождество ``D1 @ D0 = 0`` выполняется точно.
"""

import numpy as np
import scipy.sparse as sps


def grid_sizes(nx, ny, periodic=False):
    """Число узлов, x-рёбер, y-рёбер и ячеек."""
    if periodic:
        return nx * ny, nx * ny, nx * ny, nx * ny
    return (nx + 1) * (ny + 1), nx * (ny + 1), (nx + 1) * ny, nx * ny


def incidence(nx, ny, periodic=False):
    """Матрицы D0 (узлы → рёбра) и D1 (рёбра → ячейки) в формате scipy.sparse.csr."""
    n_nodes, n_ex, n_ey, n_cells = grid_sizes(nx, ny, periodic)
    mx, my = (nx, ny) if periodic else (nx + 1, ny + 1)          # число узлов по осям

    def node(i, j):
        return (i % mx) + mx * (j % my)

    def ex(i, j):                                                # x-ребро из узла (i, j)
        return (i % nx) + nx * (j % my)

    def ey(i, j):                                                # y-ребро из узла (i, j)
        return n_ex + (i % mx) + mx * (j % ny)

    rows, cols, vals = [], [], []
    for j in range(my):
        for i in range(nx):
            rows += [ex(i, j)] * 2; cols += [node(i, j), node(i + 1, j)]; vals += [-1, 1]
    for j in range(ny):
        for i in range(mx):
            rows += [ey(i, j)] * 2; cols += [node(i, j), node(i, j + 1)]; vals += [-1, 1]
    D0 = sps.csr_matrix((vals, (rows, cols)), shape=(n_ex + n_ey, n_nodes))

    rows, cols, vals = [], [], []
    for j in range(ny):
        for i in range(nx):
            c = i + nx * j
            rows += [c] * 4
            cols += [ex(i, j), ey(i + 1, j), ex(i, j + 1), ey(i, j)]
            vals += [1, 1, -1, -1]                               # обход против часовой стрелки
    D1 = sps.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_ex + n_ey))
    return D0, D1


def node_coords(nx, ny, hx=1.0, hy=1.0, periodic=False):
    """Координаты узлов в порядке их нумерации: массив формы (число узлов, 2)."""
    mx, my = (nx, ny) if periodic else (nx + 1, ny + 1)
    i, j = np.meshgrid(np.arange(mx), np.arange(my))
    return np.column_stack([i.ravel() * hx, j.ravel() * hy])


def edge_endpoints(nx, ny, hx=1.0, hy=1.0):
    """Начала и концы рёбер (без периодичности): два массива формы (число рёбер, 2)."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny + 1))
    start_x = np.column_stack([i.ravel() * hx, j.ravel() * hy])
    i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny))
    start_y = np.column_stack([i.ravel() * hx, j.ravel() * hy])
    start = np.vstack([start_x, start_y])
    step = np.vstack([np.tile([hx, 0.0], (len(start_x), 1)), np.tile([0.0, hy], (len(start_y), 1))])
    return start, start + step


def betti_numbers(nx, ny, periodic=False):
    """Числа Бетти (b0, b1, b2) клеточного комплекса сетки: размерности ker d / im d."""
    D0, D1 = incidence(nx, ny, periodic)
    r0 = np.linalg.matrix_rank(D0.toarray())
    r1 = np.linalg.matrix_rank(D1.toarray())
    n_nodes, n_ex, n_ey, n_cells = grid_sizes(nx, ny, periodic)
    return int(n_nodes - r0), int((n_ex + n_ey) - r1 - r0), int(n_cells - r1)
