import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from gmph.mechanics import (accelerations, energy, euler_lagrange, first_order_system,
                            mass_matrix, momenta, state_function)

t = sp.Symbol("t", real=True)
m, k, g, l = sp.symbols("m k g l", positive=True)


def test_oscillator_equation():
    x = sp.Function("x")(t)
    L = m * x.diff(t)**2 / 2 - k * x**2 / 2
    assert sp.simplify(euler_lagrange(L, [x], t)[0] - (m * x.diff(t, 2) + k * x)) == 0
    assert sp.simplify(accelerations(L, [x], t)[0] + k * x / m) == 0


def test_energy_and_momentum_of_central_motion():
    r, phi = sp.Function("r")(t), sp.Function("phi")(t)
    U = sp.Function("U")
    L = m / 2 * (r.diff(t)**2 + r**2 * phi.diff(t)**2) - U(r)
    p_r, p_phi = momenta(L, [r, phi], t)
    assert sp.simplify(p_phi - m * r**2 * phi.diff(t)) == 0
    T = m / 2 * (r.diff(t)**2 + r**2 * phi.diff(t)**2)
    assert sp.simplify(energy(L, [r, phi], t) - (T + U(r))) == 0
    assert mass_matrix(L, [r, phi], t) == sp.diag(m, m * r**2)


def test_double_pendulum_accelerations_solve_equations():
    a, b = sp.Function("a")(t), sp.Function("b")(t)
    L = (m * l**2 * a.diff(t)**2 + m * l**2 * b.diff(t)**2 / 2
         + m * l**2 * a.diff(t) * b.diff(t) * sp.cos(a - b)
         + 2 * m * g * l * sp.cos(a) + m * g * l * sp.cos(b))
    acc = accelerations(L, [a, b], t)
    subs = {a.diff(t, 2): acc[0], b.diff(t, 2): acc[1]}
    # simplify не справляется с тригонометрией, поэтому проверяем в случайных точках
    rng = np.random.default_rng(1)
    for _ in range(3):
        da, db, va, vb = rng.uniform(-2, 2, 4)
        point = {a.diff(t): da, b.diff(t): db}
        values = {a: va, b: vb, m: 1.3, g: 9.8, l: 0.7}
        for eq in euler_lagrange(L, [a, b], t):
            assert abs(float(eq.subs(subs).subs(point).subs(values))) < 1e-10


def test_numeric_pendulum_conserves_energy():
    phi = sp.Function("phi")(t)
    L = m * l**2 * phi.diff(t)**2 / 2 + m * g * l * sp.cos(phi)
    params = {m: 1, l: 1, g: 9.81}
    rhs = first_order_system(L, [phi], t, params)
    E = state_function(energy(L, [phi], t), [phi], t, params)
    sol = solve_ivp(rhs, (0, 10), [2.0, 0.0], rtol=1e-10, atol=1e-12)
    energies = [E(0, y) for y in sol.y.T]
    assert np.ptp(energies) < 1e-7
    # малые колебания: период близок к 2π√(l/g)
    small = solve_ivp(rhs, (0, 2 * np.pi / np.sqrt(9.81)), [1e-3, 0.0], rtol=1e-11, atol=1e-14)
    assert abs(small.y[0, -1] - 1e-3) < 1e-8


def test_legendre_transform():
    from gmph.mechanics import hamiltonian
    r, phi = sp.Function("r")(t), sp.Function("phi")(t)
    R, PHI, pr, pphi = sp.symbols("R PHI p_r p_phi", real=True)
    L = m / 2 * (r.diff(t)**2 + r**2 * phi.diff(t)**2) - k / r
    H = hamiltonian(L, [r, phi], t, [R, PHI], [pr, pphi])
    assert sp.simplify(H - (pr**2 / (2 * m) + pphi**2 / (2 * m * R**2) + k / R)) == 0
    # заряд в поле: H = (p − eA/c)²/2m + eφ
    x = sp.Function("x")(t)
    X, P = sp.symbols("X P", real=True)
    e, c = sp.symbols("e c", positive=True)
    A, phi_el = sp.Function("A"), sp.Function("varphi")
    L_charge = m * x.diff(t)**2 / 2 + e / c * A(x) * x.diff(t) - e * phi_el(x)
    H_charge = hamiltonian(L_charge, [x], t, [X], [P])
    assert sp.simplify(H_charge - ((P - e * A(X) / c)**2 / (2 * m) + e * phi_el(X))) == 0


def test_noether_identity_and_charges():
    from gmph.mechanics import lifted_derivative, noether_charge

    x, y = sp.Function("x")(t), sp.Function("y")(t)
    V = sp.Function("V")
    L = m / 2 * (x.diff(t)**2 + y.diff(t)**2) - V(x**2 + y**2)
    rotation = [-y, x]
    assert sp.simplify(lifted_derivative(L, rotation, [x, y], t)) == 0
    Lz = noether_charge(L, rotation, [x, y], t)
    assert sp.simplify(Lz - m * (x * y.diff(t) - y * x.diff(t))) == 0
    # тождество dI/dt = ξⁱ·(d/dt ∂L/∂q̇ⁱ − ∂L/∂qⁱ) + X^c L для произвольных L и ξ
    Lg = sp.Function("Lg")(x, y, x.diff(t), y.diff(t))
    xi = [sp.Function("a")(x, y, t), sp.Function("b")(x, y, t)]
    lhs = noether_charge(Lg, xi, [x, y], t).diff(t)
    rhs = (sum(c * e for c, e in zip(xi, euler_lagrange(Lg, [x, y], t)))
           + lifted_derivative(Lg, xi, [x, y], t))
    assert sp.simplify(lhs - rhs) == 0


def test_galilean_boost_is_quasi_symmetry():
    from gmph.mechanics import lifted_derivative, noether_charge

    x = sp.Function("x")(t)
    L = m * x.diff(t)**2 / 2
    xi = [t]                                              # x → x + s t
    F = m * x                                             # X^c L = m ẋ = d(m x)/dt
    assert sp.simplify(lifted_derivative(L, xi, [x], t) - F.diff(t)) == 0
    assert sp.simplify(noether_charge(L, xi, [x], t, F) - m * (x.diff(t) * t - x)) == 0
