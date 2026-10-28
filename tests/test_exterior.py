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


def test_d_squared_and_leibniz():
    from gmph.exterior import d

    xs = sp.symbols("x1:5", real=True)
    f = sp.Function("f")(*xs)
    assert d(d({(): f}, xs), xs) == {}
    alpha = {(1,): sp.Function("a")(*xs), (3,): sp.Function("b")(*xs)}
    beta = {(2, 4): sp.Function("c")(*xs)}
    assert d(d(alpha, xs), xs) == {}
    # d(α∧β) = dα∧β − α∧dβ для 1-формы α
    assert add(d(wedge(alpha, beta), xs), scale(-1, wedge(d(alpha, xs), beta)), wedge(alpha, d(beta, xs))) == {}
    x, y = xs[:2]
    assert d({(1,): -y, (2,): x}, xs) == {(1, 2): 2}


def test_pullback_polar_and_commutes_with_d():
    from gmph.exterior import d, pullback

    x, y = sp.symbols("x y", real=True)
    r, phi = sp.symbols("r phi", positive=True)
    images = (r * sp.cos(phi), r * sp.sin(phi))
    assert pullback({(1, 2): 1}, (x, y), images, (r, phi)) == {(1, 2): r}
    alpha = {(1,): x**2 * y, (2,): sp.sin(x)}
    lhs = pullback(d(alpha, (x, y)), (x, y), images, (r, phi))
    rhs = d(pullback(alpha, (x, y), images, (r, phi)), (r, phi))
    assert add(lhs, scale(-1, rhs)) == {}


def test_poincare_potential():
    from gmph.exterior import d, poincare_potential

    xs = x, y, z = sp.symbols("x y z", real=True)
    B = sp.symbols("B_x B_y B_z", real=True)
    flux = {(2, 3): B[0], (1, 3): -B[1], (1, 2): B[2]}            # однородное поле
    A = poincare_potential(flux, xs)
    assert A == {(1,): (B[1] * z - B[2] * y) / 2, (2,): (B[2] * x - B[0] * z) / 2,
                 (3,): (B[0] * y - B[1] * x) / 2}                  # A = ½ B × r
    assert d(A, xs) == flux
    # гомотопическая формула ω = dKω + Kdω для незамкнутой формы
    omega = {(1,): x * y**2, (2,): z, (3,): x**3 * y}
    total = add(d(poincare_potential(omega, xs), xs), poincare_potential(d(omega, xs), xs))
    assert add(total, scale(-1, omega)) == {}
    assert poincare_potential({(): 5}, xs) == {}


def test_integrate_form_orientation():
    from gmph.exterior import integrate_form
    x, y, t = sp.symbols("x y t", real=True)
    rotation = {(1,): -y, (2,): x}
    forward = integrate_form(rotation, (x, y), (sp.cos(t), sp.sin(t)), (t,), [(t, 0, 2 * sp.pi)])
    backward = integrate_form(rotation, (x, y), (sp.cos(t), -sp.sin(t)), (t,), [(t, 0, 2 * sp.pi)])
    assert forward == 2 * sp.pi and backward == -2 * sp.pi
    u, v = sp.symbols("u v", real=True)
    assert integrate_form({(1, 2): 1}, (x, y), (u, v), (u, v), [(u, 0, 2), (v, 0, 3)]) == 6
    assert integrate_form({(1, 2): 1}, (x, y), (u, v), (v, u), [(v, 0, 3), (u, 0, 2)]) == -6


def test_interior_and_lie():
    from gmph.exterior import d, interior, lie, pullback
    x, y, z, t = sp.symbols("x y z t", real=True)
    X3 = (x, y, z)
    assert interior((1, 2, 3), {(1, 2): 1}) == {(2,): 1, (1,): -2}        # ι_v dx∧dy = v¹dy − v²dx
    vec = (y * z, x**2, x + y)
    omega = {(1, 2): x * y, (2, 3): z, (1, 3): y**2}
    assert interior(vec, interior(vec, omega)) == {}
    # L_X ω = d/dt φ_t*ω при t = 0, φ_t(x) ≈ x + tX(x)
    moved = pullback(omega, X3, [c + t * v for c, v in zip(X3, vec)], X3)
    by_flow = {I: sp.diff(c, t).subs(t, 0) for I, c in moved.items()}
    assert add(lie(vec, omega, X3), scale(-1, by_flow)) == {}
    # L_X vol = (div X) vol
    assert lie(vec, {(1, 2, 3): 1}, X3) == {}
    assert lie((x, y, z), {(1, 2, 3): 1}, X3) == {(1, 2, 3): 3}
