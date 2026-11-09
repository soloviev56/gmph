"""Конечные элементы на треугольных сетках (статьи «Конечные элементы внешнего исчисления»
и «Численные методы теории упругости»).

Сетка задаётся массивом координат узлов ``points`` формы (N, 2) и массивом треугольников
``tri`` формы (M, 3) с номерами вершин против часовой стрелки.

Клетки и их ориентация:
- ребро — пара номеров узлов (i, j) с i < j, ориентированное от i к j;
- треугольник ориентирован против часовой стрелки.
Матрицы инцидентности ``D0`` (рёбра × узлы) и ``D1`` (треугольники × рёбра) — дискретный
дифференциал, как в модуле ``gmph.dec``: ``D1 @ D0 = 0``.

Формы Уитни (низшие элементы внешнего исчисления, λ_i — барицентрические координаты):
- 0-формы — «шапочки» λ_i (элементы Лагранжа P1);
- 1-форма ребра (i, j): w_ij = λ_i dλ_j − λ_j dλ_i (элементы Неделека первого рода);
- 2-форма треугольника T: dx∧dy / |T| (кусочно-постоянные функции).
Степени свободы — значения в узлах, интегралы по рёбрам и по треугольникам; d переводит
формы Уитни в формы Уитни с матрицами D0, D1.
"""

import numpy as np
import scipy.sparse as sps

LOCAL_EDGES = ((0, 1), (1, 2), (2, 0))


def square_mesh(n, length=1.0, pattern="right"):
    """Треугольная сетка квадрата [0, length]² из n×n клеток.

    pattern="right": каждая клетка делится диагональю на два прямоугольных треугольника;
    pattern="crisscross": клетка делится обеими диагоналями на четыре (добавляется центр).
    """
    h = length / n
    xs = np.arange(n + 1) * h
    X, Y = np.meshgrid(xs, xs)
    points = [np.column_stack([X.ravel(), Y.ravel()])]
    node = lambda i, j: i + (n + 1) * j
    tri = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
            if pattern == "right":
                tri += [(a, b, c), (a, c, d)]
            elif pattern == "crisscross":
                m = (n + 1) ** 2 + i + n * j
                tri += [(a, b, m), (b, c, m), (c, d, m), (d, a, m)]
            else:
                raise ValueError(pattern)
    if pattern == "crisscross":
        cx, cy = np.meshgrid(xs[:-1] + h / 2, xs[:-1] + h / 2)
        points.append(np.column_stack([cx.ravel(), cy.ravel()]))
    return np.vstack(points), np.array(tri)


def edges(tri):
    """Рёбра сетки и их связь с треугольниками.

    Возвращает (edge, tri_edge, sign): edge — массив (E, 2) пар узлов i < j;
    tri_edge[t, k] — номер локального ребра k = (0,1), (1,2), (2,0) треугольника t;
    sign[t, k] = +1, если ориентация ребра совпадает с обходом треугольника.
    """
    local = np.array([[t[a], t[b]] for t in tri for a, b in LOCAL_EDGES])
    key = np.sort(local, axis=1)
    edge, inverse = np.unique(key, axis=0, return_inverse=True)
    tri_edge = inverse.reshape(-1, 3)
    sign = np.where(local[:, 0] < local[:, 1], 1, -1).reshape(-1, 3)
    return edge, tri_edge, sign


def incidence(points, tri):
    """Матрицы инцидентности D0 (рёбра × узлы) и D1 (треугольники × рёбра)."""
    edge, tri_edge, sign = edges(tri)
    n_e = len(edge)
    D0 = sps.csr_matrix((np.r_[-np.ones(n_e), np.ones(n_e)],
                         (np.r_[np.arange(n_e), np.arange(n_e)], np.r_[edge[:, 0], edge[:, 1]])),
                        shape=(n_e, len(points)))
    D1 = sps.csr_matrix((sign.ravel(), (np.repeat(np.arange(len(tri)), 3), tri_edge.ravel())),
                        shape=(len(tri), n_e))
    return D0, D1


def barycentric_gradients(points, tri):
    """Площади треугольников (M,) и градиенты барицентрических координат (M, 3, 2)."""
    p = points[tri]                                   # (M, 3, 2)
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]   # 2·площадь (> 0 при обходе против часовой)
    grads = np.empty((len(tri), 3, 2))
    for k in range(3):                                # ∇λ_k = поворот противолежащей стороны / 2S
        a, b = p[:, (k + 1) % 3], p[:, (k + 2) % 3]
        grads[:, k, 0] = (a[:, 1] - b[:, 1]) / det
        grads[:, k, 1] = (b[:, 0] - a[:, 0]) / det
    return det / 2, grads


def _assemble(rows, cols, vals, shape):
    return sps.csr_matrix((np.ravel(vals), (np.ravel(rows), np.ravel(cols))), shape=shape)


def p1_matrices(points, tri):
    """Матрицы жёсткости K_ij = ∫∇λ_i·∇λ_j и масс M_ij = ∫λ_iλ_j элементов P1."""
    area, g = barycentric_gradients(points, tri)
    local_K = area[:, None, None] * np.einsum("tia,tja->tij", g, g)
    local_M = area[:, None, None] * (np.ones((3, 3)) + np.eye(3)) / 12
    rows = np.repeat(tri[:, :, None], 3, axis=2)
    cols = np.repeat(tri[:, None, :], 3, axis=1)
    n = len(points)
    return _assemble(rows, cols, local_K, (n, n)), _assemble(rows, cols, local_M, (n, n))


def whitney1_matrices(points, tri):
    """Матрицы элементов Неделека (1-форм Уитни): C_ab = ∫ rot w_a rot w_b, M_ab = ∫ w_a·w_b.

    Здесь rot w = ∂_x w_y − ∂_y w_x; для w_ij на треугольнике rot w_ij = 2 ∇λ_i × ∇λ_j.
    Знаки учитывают ориентацию рёбер: C = D1ᵀ diag(1/|T|) D1.
    """
    area, g = barycentric_gradients(points, tri)
    edge, tri_edge, sign = edges(tri)
    lam_mass = (np.ones((3, 3)) + np.eye(3)) / 12     # ∫λ_aλ_b / |T|
    M_loc = np.zeros((len(tri), 3, 3))
    for k, (i, j) in enumerate(LOCAL_EDGES):
        for l, (p, q) in enumerate(LOCAL_EDGES):
            # ∫(λ_i∇λ_j − λ_j∇λ_i)·(λ_p∇λ_q − λ_q∇λ_p)
            dot = lambda a, b: np.einsum("ta,ta->t", g[:, a], g[:, b])
            M_loc[:, k, l] = area * (lam_mass[i, p] * dot(j, q) - lam_mass[i, q] * dot(j, p)
                                     - lam_mass[j, p] * dot(i, q) + lam_mass[j, q] * dot(i, p))
    M_loc *= sign[:, :, None] * sign[:, None, :]
    rows = np.repeat(tri_edge[:, :, None], 3, axis=2)
    cols = np.repeat(tri_edge[:, None, :], 3, axis=1)
    n_e = len(edge)
    M = _assemble(rows, cols, M_loc, (n_e, n_e))
    _, D1 = incidence(points, tri)
    C = (D1.T @ sps.diags(1 / area) @ D1).tocsr()
    return C, M


def whitney1_at_centroids(points, tri, coeffs):
    """Векторы 1-формы Уитни Σ coeffs_e w_e в центрах треугольников, массив (M, 2).

    В центре λ = 1/3, поэтому w_ij = (∇λ_j − ∇λ_i)/3.
    """
    _, g = barycentric_gradients(points, tri)
    _, tri_edge, sign = edges(tri)
    v = np.zeros((len(tri), 2))
    for k, (i, j) in enumerate(LOCAL_EDGES):
        v += (sign[:, k] * coeffs[tri_edge[:, k]])[:, None] * (g[:, j] - g[:, i]) / 3
    return v


def boundary_nodes(points, length=1.0, tol=1e-12):
    """Логическая маска узлов на границе квадрата [0, length]²."""
    return (np.isclose(points, 0, atol=tol) | np.isclose(points, length, atol=tol)).any(axis=1)
