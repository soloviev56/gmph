import sympy as sp

from gmph.fields import canonical_stress, divergence, euler_lagrange, momentum_densities

t, x = sp.symbols("t x", real=True)
rho, T = sp.symbols("rho T", positive=True)
y = sp.Function("y")(t, x)
L_STRING = rho * y.diff(t)**2 / 2 - T * y.diff(x)**2 / 2


def test_string_wave_equation():
    eq = euler_lagrange(L_STRING, [y], (t, x))[0]
    assert sp.simplify(eq - (rho * y.diff(t, 2) - T * y.diff(x, 2))) == 0
    assert momentum_densities(L_STRING, [y], t) == [rho * y.diff(t)]


def test_stress_conserved_on_shell():
    Tmn = canonical_stress(L_STRING, [y], (t, x))
    assert sp.simplify(Tmn[0, 0] - (rho * y.diff(t)**2 / 2 + T * y.diff(x)**2 / 2)) == 0   # энергия
    on_shell = {y.diff(t, 2): T / rho * y.diff(x, 2)}
    assert all(sp.simplify(e.subs(on_shell)) == 0 for e in divergence(Tmn, (t, x)))
