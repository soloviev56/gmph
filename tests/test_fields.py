import sympy as sp

from gmph.fields import canonical_stress, divergence, euler_lagrange, momentum_densities  # noqa: F401

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


def test_noether_u1_current():
    from gmph.fields import noether_current

    m = sp.Symbol("m", positive=True)
    psi, psib = sp.Function("psi")(t, x), sp.Function("psib")(t, x)
    L = psi.diff(t) * psib.diff(t) - psi.diff(x) * psib.diff(x) - m**2 * psi * psib
    j = noether_current(L, [psi, psib], [sp.I * psi, -sp.I * psib], (t, x))
    on_shell = {psi.diff(t, 2): psi.diff(x, 2) - m**2 * psi, psib.diff(t, 2): psib.diff(x, 2) - m**2 * psib}
    assert sp.simplify((j[0].diff(t) + j[1].diff(x)).subs(on_shell)) == 0
