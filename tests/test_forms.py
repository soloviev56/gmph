import sympy as sp
from sympy.diffgeom import Differential, WedgeProduct

from gmph.forms import chart, components, pair, plane_polar, to_symbols


def test_dual_basis():
    X = chart("x y z", real=True)
    for i, form in enumerate(X.forms):
        for j, vec in enumerate(X.vectors):
            assert pair(form, vec) == (1 if i == j else 0)


def test_differential_components_are_partial_derivatives():
    X = chart("x y", real=True)
    x, y = X.coords
    f = x**2 + 3 * x * y
    sx, sy = X.symbols
    assert components(Differential(f), X) == [2 * sx + 3 * sy, 3 * sx]


def test_pairing_is_directional_derivative():
    X = chart("x y", real=True)
    x, y = X.coords
    ex, ey = X.vectors
    sx, sy = X.symbols
    assert pair(Differential(x**2 + 3 * x * y), 5 * ex - 2 * ey) == 4 * sx + 15 * sy


def test_to_symbols_replaces_fields():
    X = chart("x y", real=True)
    x, y = X.coords
    assert to_symbols(x * y + 1, X) == X.symbols[0] * X.symbols[1] + 1


def test_covector_components_transform_with_jacobian():
    cart, polar = plane_polar()
    x, y = cart.coords
    r, phi = polar.symbols
    f = x**2 * y
    omega_polar = components(Differential(f), polar)
    f_polar = (r * sp.cos(phi)) ** 2 * (r * sp.sin(phi))
    assert sp.simplify(omega_polar[0] - sp.diff(f_polar, r)) == 0
    assert sp.simplify(omega_polar[1] - sp.diff(f_polar, phi)) == 0


def test_area_form_in_polar_coordinates():
    cart, polar = plane_polar()
    dx, dy = cart.forms
    e_r, e_phi = polar.vectors
    assert pair(WedgeProduct(dx, dy), e_r, e_phi) == polar.symbols[0]
