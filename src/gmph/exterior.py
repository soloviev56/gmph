"""Внешняя алгебра в компонентах: формы, внешнее произведение, оператор Ходжа.

Форма хранится как словарь {возрастающий набор индексов: коэффициент};
индексы нумеруются с 1. Например, 3 e¹∧e³ − e²∧e³ — это
``{(1, 3): 3, (2, 3): -1}``, а 0-форма (число) c — ``{(): c}``.
Коэффициенты при упорядоченных индексах совпадают с компонентами
ω_{i₁…i_k} в соглашении ω = (1/k!) ω_{i₁…i_k} e^{i₁}∧…∧e^{i_k}.
"""

from itertools import combinations

import sympy as sp


def sort_with_sign(indices):
    """Упорядочить индексы; вернуть (знак перестановки, набор) или (0, None) при повторе."""
    idx, sign = list(indices), 1
    if len(set(idx)) < len(idx):
        return 0, None
    for i in range(len(idx)):
        for j in range(len(idx) - 1 - i):
            if idx[j] > idx[j + 1]:
                idx[j], idx[j + 1] = idx[j + 1], idx[j]
                sign = -sign
    return sign, tuple(idx)


def _clean(form):
    return {K: c for K, c in ((K, sp.simplify(c)) for K, c in form.items()) if c != 0}


def add(*forms):
    """Сумма форм."""
    result = {}
    for form in forms:
        for I, c in form.items():
            result[I] = result.get(I, 0) + c
    return _clean(result)


def scale(c, form):
    """Произведение формы на число."""
    return _clean({I: c * x for I, x in form.items()})


def wedge(*forms):
    """Внешнее произведение форм."""
    result = {(): 1}
    for b in forms:
        product = {}
        for I, x in result.items():
            for J, y in b.items():
                sign, K = sort_with_sign(I + J)
                if sign:
                    product[K] = product.get(K, 0) + sign * x * y
        result = _clean(product)
    return result


def d(form, coords):
    """Внешний дифференциал: dω = Σ_I Σ_j ∂_j ω_I dx^j∧dx^I.

    ``coords`` — координаты x¹, …, xⁿ (символы SymPy), от которых зависят
    компоненты; базисная 1-форма dx^j имеет индекс j (с 1).
    """
    return add(*[wedge({(j,): sp.diff(c, x)}, {I: 1})
                 for I, c in form.items() for j, x in enumerate(coords, 1)])


def interior(vector, form):
    """Внутреннее произведение ι_X ω: подстановка X в первый аргумент.

    ``vector`` — компоненты X¹, …, Xⁿ; ι_X(e^{i₁}∧…∧e^{i_k}) = Σ_m (−1)^m X^{i_m} e^{…без i_m…}.
    """
    result = {}
    for I, c in form.items():
        for m, i in enumerate(I):
            J = I[:m] + I[m + 1:]
            result[J] = result.get(J, 0) + (-1) ** m * vector[i - 1] * c
    return _clean(result)


def lie(vector, form, coords):
    """Производная Ли формы по формуле Картана: L_X ω = d ι_X ω + ι_X dω."""
    return add(d(interior(vector, form), coords), interior(vector, d(form, coords)))


def pullback(form, old_coords, images, new_coords):
    """Перенос формы назад по отображению y ↦ x(y).

    ``images`` — выражения x^i(y) через ``new_coords``; в компонентах x
    заменяется на x(y), а dx^i — на Σ_j ∂x^i/∂y^j dy^j.
    """
    subs = dict(zip(old_coords, images))
    dx = [{(j,): sp.diff(xi, y) for j, y in enumerate(new_coords, 1)} for xi in images]
    return add(*[scale(sp.sympify(c).subs(subs), wedge(*[dx[i - 1] for i in I])) for I, c in form.items()])


def integrate_form(form, coords, images, params, bounds):
    """Интеграл k-формы по параметризованной поверхности: ∫_Σ ω = ∫_D φ*ω.

    ``images`` — φ(u) через параметры ``params`` (их порядок задаёт
    ориентацию), ``bounds`` — пределы для ``sympy.integrate``:
    [(u¹, a, b), …], внутренний интеграл первым.
    """
    pulled = pullback(form, coords, images, params)
    top = tuple(range(1, len(params) + 1))
    return sp.simplify(sp.integrate(pulled.get(top, 0), *bounds))


def poincare_potential(form, coords):
    """Гомотопический оператор Пуанкаре для звёздной области с центром в нуле.

    Kω(x) = ∫₀¹ t^{k−1} (ι_x ω)(tx) dt, где ι_x — подстановка радиус-вектора
    x^i ∂_i в первый аргумент. Для любой формы ω = dKω + Kdω; поэтому при
    dω = 0 форма Kω — потенциал: d(Kω) = ω.
    """
    t = sp.Dummy("t")
    stretch = {x: t * x for x in coords}
    result = {}
    for I, c in form.items():
        k = len(I)
        coeff = sp.integrate(t ** (k - 1) * sp.sympify(c).subs(stretch, simultaneous=True), (t, 0, 1))
        for m, i in enumerate(I):
            J = I[:m] + I[m + 1:]
            result[J] = result.get(J, 0) + (-1) ** m * coords[i - 1] * coeff
    return _clean(result)


def value(form, *vectors):
    """Значение k-формы на k векторах: e^I(v_1, …, v_k) = det[v_j^{i_m}]."""
    total = 0
    for I, c in form.items():
        minor = sp.Matrix(len(I), len(I), lambda m, j: vectors[j][I[m] - 1])
        total += c * minor.det()
    return sp.simplify(total)


def raise_indices(form, g):
    """Контравариантные компоненты α^I = Σ_K det(g⁻¹[I, K]) α_K (I, K возрастающие)."""
    g_inv = sp.Matrix(g).inv()
    n = g_inv.rows
    result = {}
    for K, c in form.items():
        for I in combinations(range(1, n + 1), len(K)):
            minor = g_inv.extract([i - 1 for i in I], [k - 1 for k in K]) if K else sp.Matrix([[1]])
            result[I] = result.get(I, 0) + minor.det() * c
    return _clean(result)


def inner(alpha, beta, g):
    """Скалярное произведение k-форм, порождённое метрикой: ⟨α, β⟩ = Σ_I α^I β_I."""
    up = raise_indices(alpha, g)
    return sp.simplify(sum((up.get(I, 0) * c for I, c in beta.items()), sp.S(0)))


def volume(g):
    """Форма объёма √|det g| e¹∧…∧eⁿ."""
    g = sp.Matrix(g)
    return {tuple(range(1, g.rows + 1)): sp.sqrt(sp.Abs(g.det()))}


def hodge(form, g):
    """Оператор Ходжа: α∧⋆β = ⟨α, β⟩ vol.

    В компонентах (⋆α)_J = √|det g| Σ_I α^I ε_{IJ}, где J — дополнение I,
    а ε_{IJ} — знак перестановки (I, J) набора (1, …, n).
    """
    g = sp.Matrix(g)
    n = g.rows
    root = sp.sqrt(sp.Abs(g.det()))
    result = {}
    for I, c in raise_indices(form, g).items():
        J = tuple(i for i in range(1, n + 1) if i not in I)
        sign, _ = sort_with_sign(I + J)
        result[J] = result.get(J, 0) + sign * root * c
    return _clean(result)


def show(form, names=None):
    """Запись формы строкой: names — обозначения базисных 1-форм (по умолчанию e¹, e², …)."""
    if not form:
        return "0"
    sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    name = (lambda i: names[i - 1]) if names else (lambda i: "e" + str(i).translate(sup))
    terms = []
    for I, c in sorted(form.items(), key=lambda item: (len(item[0]), item[0])):
        basis = "∧".join(name(i) for i in I)
        terms.append(f"({c})" + (f"·{basis}" if basis else ""))
    return " + ".join(terms)


def _signature_sign(g):
    """sign(det g). Знак постоянен на всей карте, поэтому если SymPy не может
    определить его символьно (например, det = r² sin²θ), он берётся в
    произвольной точке общего положения."""
    det = sp.Matrix(g).det()
    s = sp.sign(det)
    if s in (1, -1):
        return s
    point = {sym: sp.Rational(7, 10) + sp.Rational(13, 100) * i for i, sym in enumerate(sorted(det.free_symbols, key=str))}
    return sp.sign(sp.N(det.subs(point)))


def codifferential(form, coords, g):
    """Кодифференциал δ — формально сопряжённый к d оператор (конвенции).

    На k-формах в n измерениях δ = (−1)^{n(k+1)+1} s ⋆d⋆, где s = sign det g.
    Форма должна быть однородной степени k.
    """
    g = sp.Matrix(g)
    degrees = {len(I) for I in form}
    if not degrees:
        return {}
    assert len(degrees) == 1, "δ определён для однородных форм"
    k, n = degrees.pop(), g.rows
    if k == 0:
        return {}
    return scale((-1)**(n * (k + 1) + 1) * _signature_sign(g), hodge(d(hodge(form, g), coords), g))


def laplace_de_rham(form, coords, g):
    """Оператор Лапласа–де Рама Δ = dδ + δd (на однородной форме)."""
    return add(d(codifferential(form, coords, g), coords), codifferential(d(form, coords), coords, g))
