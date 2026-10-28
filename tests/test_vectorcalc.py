import sympy as sp

from gmph.vectorcalc import div, from_physical, grad, laplacian, physical, rot

x, y, z = XYZ = sp.symbols("x y z", real=True)
r, th, ph = SPH = sp.symbols("r theta phi", positive=True)
G_SPH = sp.diag(1, r**2, r**2 * sp.sin(th)**2)


def cartesian_rot(v):
    return [sp.diff(v[2], y) - sp.diff(v[1], z),
            sp.diff(v[0], z) - sp.diff(v[2], x),
            sp.diff(v[1], x) - sp.diff(v[0], y)]


def test_cartesian_operators():
    f = x**2 * y + sp.sin(z)
    v = [x * y, y * z**2, x**3]
    assert grad(f, XYZ, sp.eye(3)) == [2 * x * y, x**2, sp.cos(z)]
    assert [sp.simplify(a - b) for a, b in zip(rot(v, XYZ, sp.eye(3)), cartesian_rot(v))] == [0, 0, 0]
    assert sp.simplify(div(v, XYZ, sp.eye(3)) - (y + z**2)) == 0
    assert sp.simplify(laplacian(f, XYZ, sp.eye(3)) - (2 * y - sp.sin(z))) == 0


def test_spherical_textbook_formulas():
    f = sp.Function("f")(*SPH)
    textbook = (sp.diff(r**2 * f.diff(r), r) / r**2
                + sp.diff(sp.sin(th) * f.diff(th), th) / (r**2 * sp.sin(th))
                + f.diff(ph, 2) / (r**2 * sp.sin(th)**2))
    assert sp.simplify(laplacian(f, SPH, G_SPH) - textbook) == 0
    # поле точечного заряда e_r / r²: div = 0, rot = 0 при r ≠ 0
    coulomb = from_physical([1 / r**2, 0, 0], G_SPH)
    assert div(coulomb, SPH, G_SPH) == 0
    assert rot(coulomb, SPH, G_SPH) == [0, 0, 0]


def test_physical_components_roundtrip():
    v_hat = [sp.Function(n)(*SPH) for n in ("a", "b", "c")]
    assert physical(from_physical(v_hat, G_SPH), G_SPH) == v_hat
    # rot (r sin θ e_φ) = 2 e_z = 2 cos θ e_r − 2 sin θ e_θ (вращение с угловой скоростью 1)
    rigid = from_physical([0, 0, r * sp.sin(th)], G_SPH)
    w = physical(rot(rigid, SPH, G_SPH), G_SPH)
    assert [sp.simplify(a - b) for a, b in zip(w, [2 * sp.cos(th), -2 * sp.sin(th), 0])] == [0, 0, 0]


def test_physical_components_have_no_abs():
    assert from_physical([0, 0, 1], G_SPH) == [0, 0, 1 / (r * sp.sin(th))]
