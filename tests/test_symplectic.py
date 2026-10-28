import sympy as sp

from gmph.exterior import add, d, interior
from gmph.symplectic import canonical_omega, form_matrix, hamiltonian_field, poisson

q, p, m, k = sp.symbols("q p m k", real=True)


def test_matches_conventions():
    omega = canonical_omega(1)
    assert form_matrix(omega, 2) == sp.Matrix([[0, -1], [1, 0]])
    H = p**2 / (2 * m) + k * q**2 / 2
    assert hamiltonian_field(H, (q, p), omega) == [p / m, -k * q]
    assert poisson(q, p, (q, p), omega) == 1


def test_field_solves_defining_equation():
    th, ph, S, g = sp.symbols("theta phi S gamma", positive=True)
    omega = {(1, 2): -S * sp.sin(th)}                    # спин: dS_z∧dφ
    H = -g * S * sp.cos(th)
    X = hamiltonian_field(H, (th, ph), omega)
    assert X == [0, -g]
    assert add(interior(X, omega), d({(): H}, (th, ph))) == {}


def test_poisson_two_degrees():
    q1, q2, p1, p2 = Z = sp.symbols("q1 q2 p1 p2", real=True)
    omega = canonical_omega(2)
    Lz = q1 * p2 - q2 * p1
    assert poisson(q1, Lz, Z, omega) == -q2
    assert poisson(Lz, p1**2 + p2**2, Z, omega) == 0
