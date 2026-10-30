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


# --- Многообразия: стереографические карты и коммутатор полей ----------------

def test_stereographic_transition():
    u1, u2 = sp.symbols("u1 u2", real=True)
    r2 = u1**2 + u2**2
    X = sp.Matrix([2 * u1, 2 * u2, r2 - 1]) / (r2 + 1)       # обратная к u = (x, y)/(1 − z)
    assert sp.simplify(X.dot(X) - 1) == 0
    assert sp.simplify(X[0] / (1 - X[2]) - u1) == 0
    v = sp.Matrix([X[0], X[1]]) / (1 + X[2])
    assert sp.simplify(v - sp.Matrix([u1, u2]) / r2) == sp.zeros(2, 1)   # v = u/|u|²
    assert sp.simplify(v.jacobian([u1, u2]).det() + 1 / r2**2) == 0       # якобиан −1/|u|⁴


def test_linear_field_bracket_sign():
    xs = sp.symbols("x1:4", real=True)

    def field(A):
        return A * sp.Matrix(xs)

    def bracket(F, G):
        return sp.Matrix([sum(F[j] * sp.diff(G[i], xs[j]) - G[j] * sp.diff(F[i], xs[j])
                              for j in range(3)) for i in range(3)])

    A = sp.Matrix(3, 3, sp.symbols("a0:9"))
    B = sp.Matrix(3, 3, sp.symbols("b0:9"))
    assert sp.expand(bracket(field(A), field(B)) + field(A * B - B * A)) == sp.zeros(3, 1)
    Jx = sp.Matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    Jy = sp.Matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    Jz = sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    assert Jx * Jy - Jy * Jx == Jz
    x, y, z = xs
    assert field(Jx) == sp.Matrix([0, -z, y])                             # L_x = y∂_z − z∂_y
    assert bracket(field(Jx), field(Jy)) == -field(Jz)                    # [L_x, L_y] = −L_z


# --- Термодинамика: dU = δQ − p dV, δQ = T dS --------------------------------

def test_ideal_gas_entropy_form():
    from gmph.exterior import d, scale

    T, V, nu, R, Cv = sp.symbols("T V nu R C_V", positive=True)
    p = nu * R * T / V
    heat = {(1,): Cv, (2,): p}                    # δQ = dU + p dV, U = C_V T; индексы 1, 2 ↔ dT, dV
    assert d(heat, (T, V)) == {(1, 2): nu * R / V}               # δQ не замкнута
    assert d(scale(1 / T, heat), (T, V)) == {}                   # δQ/T = dS


def test_symplectic_matrix_form():
    n = 2
    qs, ps = sp.symbols("q1:3"), sp.symbols("p1:3")
    Omega = sp.Matrix(sp.BlockMatrix([[sp.zeros(n), -sp.eye(n)], [sp.eye(n), sp.zeros(n)]]))
    u = sp.Matrix(sp.symbols("u1:5"))
    v = sp.Matrix(sp.symbols("v1:5"))
    # ω = Σ dp_i∧dq^i при z = (q, p): ω(u, v) = Σ (u_{p_i} v_{q_i} − u_{q_i} v_{p_i})
    omega_uv = sum(u[n + i] * v[i] - u[i] * v[n + i] for i in range(n))
    assert sp.expand((u.T * Omega * v)[0] - omega_uv) == 0
    assert Omega.inv() == -Omega
    H = sp.Function("H")(*qs, *ps)
    grad = sp.Matrix([H.diff(c) for c in (*qs, *ps)])
    zdot = Omega.inv() * grad                         # ż = Ω⁻¹∇H
    assert all(zdot[i] == H.diff(ps[i]) and zdot[n + i] == -H.diff(qs[i]) for i in range(n))


def test_spin_symplectic_form():
    from gmph.symplectic import hamiltonian_field, poisson

    th, ph = sp.symbols("theta phi", real=True)
    S, gamma = sp.symbols("S gamma", positive=True)
    omega_spin = {(1, 2): -S * sp.sin(th)}            # ω = dS_z∧dφ в координатах (θ, φ)
    Svec = sp.Matrix([S * sp.sin(th) * sp.cos(ph), S * sp.sin(th) * sp.sin(ph), S * sp.cos(th)])
    pb = lambda f, g: poisson(f, g, (th, ph), omega_spin)
    for i in range(3):                                # {S_x, S_y} = S_z и циклически
        assert sp.simplify(pb(Svec[i], Svec[(i + 1) % 3]) - Svec[(i + 2) % 3]) == 0
    Bvec = sp.Matrix(sp.symbols("B_x B_y B_z", real=True))
    X = hamiltonian_field(-gamma * Bvec.dot(Svec), (th, ph), omega_spin)
    Sdot = Svec.jacobian([th, ph]) * sp.Matrix(X)
    assert sp.simplify(Sdot - gamma * Svec.cross(Bvec)) == sp.zeros(3, 1)
    Bz = sp.Symbol("B", positive=True)
    assert hamiltonian_field(-gamma * Bz * Svec[2], (th, ph), omega_spin) == [0, -gamma * Bz]


# --- Электростатика формами: dℰ = 0, d𝒟 = 4πρ vol, 𝒟 = ⋆ℰ, ℰ = −dφ ----------

def test_electrostatics_forms():
    from gmph.exterior import d, hodge

    x, y, z = X = sp.symbols("x y z", real=True)
    rho0 = sp.Symbol("rho_0", positive=True)
    # внутри однородно заряженного шара φ = −(2π/3) ρ₀ r² + const
    phi = -sp.Rational(2, 3) * sp.pi * rho0 * (x**2 + y**2 + z**2)
    E_form = {(i,): -sp.diff(phi, c) for i, c in enumerate(X, 1)}
    D_form = hodge(E_form, sp.eye(3))
    assert d(E_form, X) == {}
    assert d(D_form, X) == {(1, 2, 3): 4 * sp.pi * rho0}
    # 𝒟 = D·dS: компонента при dy∧dz — D_x
    assert D_form[(2, 3)] == E_form[(1,)]


# --- Магнитостатика формами: dℬ = 0, dℋ = (4π/c)𝒥, ℬ = ⋆ℋ = d𝒜 ---------------

def test_magnetostatics_forms():
    from gmph.exterior import d, hodge

    x, y, z = X = sp.symbols("x y z", real=True)
    j0, c = sp.symbols("j_0 c", positive=True)
    # внутри провода с током j₀ вдоль z: H = (2π j₀/c)(−y, x, 0)
    H_form = {(1,): -2 * sp.pi * j0 / c * y, (2,): 2 * sp.pi * j0 / c * x}
    assert d(H_form, X) == {(1, 2): 4 * sp.pi / c * j0}               # 𝒥 = j₀ dx∧dy
    B_form = hodge(H_form, sp.eye(3))
    assert d(B_form, X) == {}
    A_form = {(3,): -sp.pi * j0 / c * (x**2 + y**2)}                  # A = A_z e_z
    assert d(A_form, X) == B_form


def test_maxwell_forms_3plus1():
    from gmph.exterior import add, d, hodge, scale, wedge

    X = sp.symbols("x0:4", real=True)
    c = sp.Symbol("c", positive=True)
    eta = sp.diag(1, -1, -1, -1)

    def fields(name):
        return [sp.Function(f"{name}_{s}")(*X) for s in "xyz"]

    def one(v):                       # v·dr; индексы 1…4 ↔ x⁰…x³
        return {(2,): v[0], (3,): v[1], (4,): v[2]}

    def two(v):                       # v·dS
        return {(3, 4): v[0], (2, 4): -v[1], (2, 3): v[2]}

    E, B, D, H, J = (fields(n) for n in "EBDHj")
    rho = sp.Function("rho")(*X)
    dx0 = {(1,): 1}
    F = add(wedge(dx0, one(E)), scale(-1, two(B)))
    # совпадает с компонентами F_{0i} = E^i, F_{12} = −B_z, …
    assert F == {(1, 2): E[0], (1, 3): E[1], (1, 4): E[2], (2, 3): -B[2], (2, 4): B[1], (3, 4): -B[0]}
    # в пустоте G = −⋆F
    assert add(hodge(F, eta), two(E), wedge(dx0, one(B))) == {}
    # ⋆j = c𝔍 при j_μ = (cρ, −j)
    frak_J = add({(2, 3, 4): rho}, scale(-1 / c, wedge(dx0, two(J))))
    j_lo = {(1,): c * rho, (2,): -J[0], (3,): -J[1], (4,): -J[2]}
    assert add(hodge(j_lo, eta), scale(-c, frak_J)) == {}
    # dG = 4π𝔍 ⇔ div D = 4πρ и rot H = (1/c)∂_t D + (4π/c) j
    G = add(two(D), wedge(dx0, one(H)))
    dG = d(G, X)
    assert sp.simplify(dG[(2, 3, 4)] - sum(sp.diff(D[i], X[i + 1]) for i in range(3))) == 0
    rot_H_x = sp.diff(H[2], X[2]) - sp.diff(H[1], X[3])
    assert sp.simplify(dG[(1, 3, 4)] - (sp.diff(D[0], X[0]) - rot_H_x)) == 0
    assert sp.simplify(frak_J[(1, 3, 4)] + J[0] / c) == 0


# --- Преобразование Фурье: f̃(k) = ∫ f e^{−ikx} dx, по времени e^{+iωt} ---------

def test_fourier_conventions():
    from gmph.fourier import fourier, fourier_time, inverse_fourier, inverse_fourier_time

    x, k, t, w = sp.symbols("x k t omega", real=True)
    a = sp.Symbol("a", positive=True)
    gauss = sp.exp(-x**2 / 2)
    G = fourier(gauss, x, k)
    assert sp.simplify(G - sp.sqrt(2 * sp.pi) * sp.exp(-k**2 / 2)) == 0
    assert sp.simplify(inverse_fourier(G, k, x) - gauss) == 0
    shifted = sp.exp(-(x - a)**2 / 2)                       # сдвиг на +a даёт множитель e^{−ika}
    assert sp.simplify(fourier(shifted, x, k) - sp.exp(-sp.I * k * a) * G) == 0
    assert sp.simplify(fourier(gauss.diff(x), x, k) - sp.I * k * G) == 0
    # по времени: затухающее колебание θ(t) e^{−at} → 1/(a − iω), и обратно
    decay = sp.exp(-a * t) * sp.Heaviside(t)
    Fw = fourier_time(decay, t, w)
    assert sp.simplify(Fw - 1 / (a - sp.I * w)) == 0
    assert sp.simplify(fourier_time(sp.exp(-t**2 / 2).diff(t), t, w) + sp.I * w * fourier_time(sp.exp(-t**2 / 2), t, w)) == 0
    assert sp.simplify(inverse_fourier_time(sp.sqrt(2 * sp.pi) * sp.exp(-w**2 / 2), w, t) - sp.exp(-t**2 / 2)) == 0
