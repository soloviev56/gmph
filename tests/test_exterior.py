import random
from itertools import combinations

import sympy as sp
from sympy.diffgeom import WedgeProduct

from gmph.exterior import add, hodge, inner, scale, show, volume, wedge, value
from gmph.forms import chart, pair


def random_form(n, k, rng):
    return {I: sp.Integer(rng.randint(-3, 3)) for I in combinations(range(1, n + 1), k)}


def test_wedge_basics():
    a = {(1,): 1, (2,): 2}
    b = {(1,): 3, (2,): -1}
    assert wedge(a, b) == {(1, 2): -7}
    assert wedge(a, a) == {}
    omega = {(1, 2): 1, (3, 4): 1}
    assert wedge(omega, omega) == {(1, 2, 3, 4): 2}


def test_wedge_matches_sympy_diffgeom():
    rng = random.Random(3)
    X = chart("x y z w", real=True)
    d, e = X.forms, X.vectors

    def to_sympy(form):
        return sum((c * (WedgeProduct(*[d[i - 1] for i in I]) if len(I) > 1 else d[I[0] - 1])
                    for I, c in form.items()), sp.S(0))

    a, c = random_form(4, 1, rng), random_form(4, 2, rng)
    vs = [[sp.Integer(rng.randint(-2, 2)) for _ in range(4)] for _ in range(3)]
    V = [sum(x * e[i] for i, x in enumerate(v)) for v in vs]
    assert value(wedge(c, a), *vs) == pair(WedgeProduct(to_sympy(c), to_sympy(a)), *V)


def test_hodge_euclidean_r3():
    g = sp.eye(3)
    assert hodge({(1,): 1}, g) == {(2, 3): 1}
    assert hodge({(2,): 1}, g) == {(1, 3): -1}          # ⋆dy = dz∧dx
    assert hodge({(1, 2): 1}, g) == {(3,): 1}
    assert hodge({(): 1}, g) == {(1, 2, 3): 1}
    a, b = sp.Matrix(sp.symbols("a1:4")), sp.Matrix(sp.symbols("b1:4"))
    star = hodge(wedge({(i,): a[i - 1] for i in (1, 2, 3)}, {(i,): b[i - 1] for i in (1, 2, 3)}), g)
    assert [sp.expand(star.get((i,), 0)) for i in (1, 2, 3)] == list(sp.expand(a.cross(b)))


def test_hodge_defining_identity_general_metric():
    rng = random.Random(5)
    g = sp.Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 1]])
    for k in range(4):
        a, b = random_form(3, k, rng), random_form(3, k, rng)
        assert wedge(a, hodge(b, g)) == scale(inner(a, b, g), volume(g))


def test_hodge_squared_signs():
    for g, s in [(sp.eye(2), 1), (sp.eye(3), 1), (sp.eye(4), 1), (sp.diag(1, -1, -1, -1), -1)]:
        n = g.rows
        for k in range(n + 1):
            for I in combinations(range(1, n + 1), k):
                form = {I: 1}
                assert hodge(hodge(form, g), g) == scale((-1) ** (k * (n - k)) * s, form)


def test_show():
    assert show({(1, 2): -7}) == "(-7)·e¹∧e²"
    assert show(add({(1,): 1}, {(1,): -1})) == "0"
