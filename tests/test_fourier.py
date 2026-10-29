import sympy as sp

from gmph.fourier import fourier, series_coefficient


def test_convolution_and_parseval():
    x, k, y = sp.symbols("x k y", real=True)
    f = sp.exp(-x**2 / 2)
    g = sp.exp(-x**2)
    conv = sp.integrate(f.subs(x, y) * g.subs(x, x - y), (y, -sp.oo, sp.oo))
    assert sp.simplify(fourier(conv, x, k) - fourier(f, x, k) * fourier(g, x, k)) == 0
    lhs = sp.integrate(f**2, (x, -sp.oo, sp.oo))
    rhs = sp.integrate(fourier(f, x, k)**2, (k, -sp.oo, sp.oo)) / (2 * sp.pi)
    assert sp.simplify(lhs - rhs) == 0


def test_series_coefficients_of_sawtooth():
    x = sp.Symbol("x", real=True)
    n = sp.Symbol("n", integer=True, nonzero=True)
    c = sp.simplify(series_coefficient(x, x, n, 2 * sp.pi, start=-sp.pi))
    assert sp.simplify(c - sp.I * (-1)**n / n) == 0
