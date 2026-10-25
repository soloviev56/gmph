"""Проверка соглашений о знаках со страницы conventions.qmd.

Если меняются конвенции, сначала меняется этот файл: тесты фиксируют
формулы страницы «Конвенции» в исполняемом виде.
"""

import sympy as sp

# --- Гамильтонова механика: ω = dp∧dq, ι_{X_H} ω = −dH --------------------

q, p = sp.symbols("q p")


def hamiltonian_field(h):
    """Компоненты (X^p, X^q) поля X_h из уравнения ι_X(dp∧dq) = −dh."""
    return (-sp.diff(h, q), sp.diff(h, p))


def omega(u, w):
    """(dp∧dq)(u, w) для векторов с компонентами (u^p, u^q)."""
    return u[0] * w[1] - u[1] * w[0]


def test_hamilton_equations():
    H = sp.Function("H")(q, p)
    Xp, Xq = hamiltonian_field(H)
    # ι_X(dp∧dq) = X^p dq − X^q dp должно равняться −dH
    assert sp.simplify(-Xq + sp.diff(H, p)) == 0
    assert sp.simplify(Xp + sp.diff(H, q)) == 0
    # q̇ = ∂H/∂p, ṗ = −∂H/∂q
    assert Xq == sp.diff(H, p) and Xp == -sp.diff(H, q)


def test_poisson_bracket():
    f, g = sp.Function("f")(q, p), sp.Function("g")(q, p)
    bracket = sp.diff(f, q) * sp.diff(g, p) - sp.diff(f, p) * sp.diff(g, q)
    Xg = hamiltonian_field(g)
    assert sp.simplify(bracket - (Xg[0] * sp.diff(f, p) + Xg[1] * sp.diff(f, q))) == 0
    assert sp.simplify(bracket - omega(Xg, hamiltonian_field(f))) == 0


# --- Электродинамика: (+,−,−,−), гауссовы единицы, как у Ландау–Лифшица -----

t, x, y, z, c, e, hbar = sp.symbols("t x y z c e hbar", positive=True)
SPACE = (x, y, z)
ETA = sp.diag(1, -1, -1, -1)
phi = sp.Function("phi")(t, x, y, z)
A_vec = [sp.Function(n)(t, x, y, z) for n in ("A_x", "A_y", "A_z")]
A_lo = ETA * sp.Matrix([phi, *A_vec])  # A_μ = (φ, −A)


def d(f, mu):
    """∂_μ при x^μ = (ct, x, y, z)."""
    return sp.diff(f, t) / c if mu == 0 else sp.diff(f, SPACE[mu - 1])


def field_tensor_up():
    F_lo = sp.Matrix(4, 4, lambda m, n: d(A_lo[n], m) - d(A_lo[m], n))
    return ETA * F_lo * ETA


E = [-sp.diff(phi, v) - sp.diff(A_vec[i], t) / c for i, v in enumerate(SPACE)]
B_z = sp.diff(A_vec[1], x) - sp.diff(A_vec[0], y)


def test_field_tensor_components():
    F = field_tensor_up()
    assert all(sp.simplify(F[0, i + 1] + E[i]) == 0 for i in range(3))
    assert sp.simplify(F[1, 2] + B_z) == 0


def test_gauss_law_sign():
    # ∂_ν F^{0ν} = −(4π/c) j^0 = −4πρ  ⇔  div E = 4πρ
    F = field_tensor_up()
    div_E = sum(sp.diff(E[i], v) for i, v in enumerate(SPACE))
    assert sp.simplify(sum(d(F[0, n], n) for n in range(4)) + div_E) == 0


psi = sp.Function("psi")(t, x, y, z)


def D(mu, f, A=A_lo):
    """D_μ = ∂_μ + (ie/ħc) A_μ."""
    return d(f, mu) + sp.I * e / (hbar * c) * A[mu] * f


def test_covariant_derivative_is_kinetic_momentum():
    # i ħ c D_0 ψ = (iħ ∂_t − eφ) ψ
    assert sp.simplify(sp.I * hbar * c * D(0, psi) - (sp.I * hbar * sp.diff(psi, t) - e * phi * psi)) == 0
    # iħ D^i ψ = (−iħ∇ − (e/c)A)^i ψ; D^i = −D_i
    for i, v in enumerate(SPACE):
        lhs = sp.I * hbar * (-D(i + 1, psi))
        rhs = -sp.I * hbar * sp.diff(psi, v) - e / c * A_vec[i] * psi
        assert sp.simplify(lhs - rhs) == 0


def test_abelian_gauge_transformation():
    f = sp.Function("f")(t, x, y, z)
    U = sp.exp(sp.I * e * f / (hbar * c))
    A_new = sp.Matrix([A_lo[m] - d(f, m) for m in range(4)])
    for m in range(4):
        assert sp.simplify(D(m, U * psi, A_new) - U * D(m, psi)) == 0


# --- Неабелева связность: D = ∂ + i𝒜, 𝒜 → U𝒜U⁻¹ + i(∂U)U⁻¹ ----------------

u, v = sp.symbols("u v")
UV = (u, v)


def matrix_field(name):
    return sp.Matrix(2, 2, lambda i, j: sp.Function(f"{name}{i}{j}")(u, v))


def test_nonabelian_curvature_and_gauge():
    A = [matrix_field("a"), matrix_field("b")]
    chi = sp.Matrix([sp.Function("chi0")(u, v), sp.Function("chi1")(u, v)])

    def Dm(m, w, conn):
        return sp.diff(w, UV[m]) + sp.I * conn[m] * w

    F = sp.diff(A[1], u) - sp.diff(A[0], v) + sp.I * (A[0] * A[1] - A[1] * A[0])
    commutator = Dm(0, Dm(1, chi, A), A) - Dm(1, Dm(0, chi, A), A)
    assert sp.simplify(commutator - sp.I * F * chi) == sp.zeros(2, 1)

    U = matrix_field("g")
    A_new = [U * A[m] * U.inv() + sp.I * sp.diff(U, UV[m]) * U.inv() for m in range(2)]
    for m in range(2):
        assert sp.simplify(Dm(m, U * chi, A_new) - U * Dm(m, chi, A)) == sp.zeros(2, 1)


# --- Оператор Ходжа: α∧⋆β = ⟨α, β⟩ vol ---------------------------------------

def test_hodge_signs():
    from gmph.exterior import hodge, scale

    assert hodge({(1,): 1}, sp.eye(3)) == {(2, 3): 1}            # ⋆dx = dy∧dz
    assert hodge({(1, 2): 1}, sp.eye(3)) == {(3,): 1}            # ⋆dx∧dy = dz
    eta = sp.diag(1, -1, -1, -1)
    two_form = {(1, 2): 1}                                       # dx⁰∧dx¹
    assert hodge(hodge(two_form, eta), eta) == scale(-1, two_form)


def test_dual_field_tensor():
    from gmph.exterior import hodge, inner, wedge

    eta = sp.diag(1, -1, -1, -1)
    Ex, Ey, Ez, Bx, By, Bz = sp.symbols("E_x E_y E_z B_x B_y B_z", real=True)

    def field(E, B):
        # F_{0i} = E^i, F_{12} = −B_z, F_{13} = B_y, F_{23} = −B_x; индексы 1…4 ↔ x⁰…x³
        return {(1, 2): E[0], (1, 3): E[1], (1, 4): E[2],
                (2, 3): -B[2], (2, 4): B[1], (3, 4): -B[0]}

    E, B = (Ex, Ey, Ez), (Bx, By, Bz)
    F = field(E, B)
    assert hodge(F, eta) == field([-b for b in B], E)
    vol = (1, 2, 3, 4)
    assert sp.expand(wedge(F, hodge(F, eta))[vol] - (Bx**2 + By**2 + Bz**2 - Ex**2 - Ey**2 - Ez**2)) == 0
    assert sp.expand(wedge(F, F)[vol] + 2 * (Ex * Bx + Ey * By + Ez * Bz)) == 0
    assert sp.expand(inner(F, F, eta) - (Bx**2 + By**2 + Bz**2 - Ex**2 - Ey**2 - Ez**2)) == 0


# --- Эрмитово произведение: антилинейно по первому аргументу ------------------

def test_hermitian_product_realification():
    x1, y1, x2, y2, c = sp.symbols("x1 y1 x2 y2 c", real=True)
    u, v = x1 + sp.I * y1, x2 + sp.I * y2

    def h(a, b):
        return sp.expand(sp.conjugate(a) * b)

    assert sp.expand(h(u, (1 + sp.I) * v) - (1 + sp.I) * h(u, v)) == 0        # линейно по второму
    assert sp.expand(h((1 + sp.I) * u, v) - (1 - sp.I) * h(u, v)) == 0        # антилинейно по первому
    assert sp.expand(sp.re(h(u, v)) - (x1 * x2 + y1 * y2)) == 0                # Re = евклидова метрика
    assert sp.expand(sp.im(h(u, v)) - (x1 * y2 - y1 * x2)) == 0                # Im = (dx∧dy)(u, v)
    assert sp.expand(sp.im(h(u, sp.I * v)) - sp.re(h(u, v))) == 0              # g(u, v) = Im⟨u|Jv⟩
