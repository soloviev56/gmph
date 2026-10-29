"""Преобразование и ряды Фурье в соглашениях пособия (страница «Конвенции»).

По координате: f̃(k) = ∫ f(x) e^{−ikx} dx, f(x) = ∫ f̃(k) e^{ikx} dk/2π.
По времени:    f̃(ω) = ∫ f(t) e^{iωt} dt,  f(t) = ∫ f̃(ω) e^{−iωt} dω/2π.
SymPy использует ядро e^{−2πikx}, поэтому аргумент пересчитывается: k → k/2π.
"""

import sympy as sp


def fourier(f, x, k):
    """f̃(k) = ∫ f(x) e^{−ikx} dx."""
    return sp.fourier_transform(f, x, k / (2 * sp.pi))


def inverse_fourier(F, k, x):
    """f(x) = ∫ F(k) e^{ikx} dk/2π."""
    return sp.inverse_fourier_transform(F, k, x / (2 * sp.pi)) / (2 * sp.pi)


def fourier_time(f, t, w):
    """f̃(ω) = ∫ f(t) e^{iωt} dt."""
    return sp.fourier_transform(f, t, -w / (2 * sp.pi))


def inverse_fourier_time(F, w, t):
    """f(t) = ∫ F(ω) e^{−iωt} dω/2π."""
    return sp.inverse_fourier_transform(F, w, -t / (2 * sp.pi)) / (2 * sp.pi)


def series_coefficient(f, x, n, L, start=0):
    """c_n = (1/L) ∫_{start}^{start+L} f(x) e^{−2πinx/L} dx."""
    return sp.integrate(f * sp.exp(-2 * sp.pi * sp.I * n * x / L), (x, start, start + L)) / L
