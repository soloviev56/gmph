"""Векторный анализ в произвольных координатах через d, ⋆, ♭ и ♯.

Векторное поле — список контравариантных компонент vⁱ (по базису ∂ᵢ),
метрика — матрица g_ij в тех же координатах, 1-формы и 2-формы — словари
из ``gmph.exterior``. Формулы:

    grad f = (df)♯,   rot v = (⋆d v♭)♯,   div v = ⋆d⋆v♭,   Δf = ⋆d⋆df.

``rot`` имеет смысл только в трёхмерном ориентированном пространстве.
Функции предполагают, что карта ориентирована положительно и что в её
области выражение под корнем √det g положительно: модули, которые
оставляет ``hodge`` (например, |sin θ| в сферической карте), снимаются.
"""

import sympy as sp

from gmph.exterior import d, hodge


def _no_abs(expr):
    return sp.simplify(sp.sympify(expr).replace(sp.Abs, lambda a: a))


def _star(form, g):
    """⋆ в карте, где √|det g| записывается без модуля."""
    return {I: _no_abs(c) for I, c in hodge(form, g).items()}


def flat(v, g):
    """Опускание индекса: v ↦ v♭ = g_ij v^j dx^i."""
    g = sp.Matrix(g)
    lowered = [sp.simplify(c) for c in g * sp.Matrix(v)]
    return {(i + 1,): c for i, c in enumerate(lowered) if c != 0}


def sharp(form, g):
    """Подъём индекса: ω ↦ ω♯ с компонентами g^ij ω_j."""
    g = sp.Matrix(g)
    omega = sp.Matrix([form.get((i + 1,), 0) for i in range(g.rows)])
    return [sp.simplify(c) for c in g.inv() * omega]


def grad(f, coords, g):
    """Градиент (df)♯ — контравариантные компоненты."""
    return sharp(d({(): f}, coords), g)


def rot(v, coords, g):
    """Ротор (⋆d v♭)♯ в трёхмерном пространстве."""
    return sharp(_star(d(flat(v, g), coords), g), g)


def div(v, coords, g):
    """Дивергенция ⋆d⋆v♭."""
    return sp.simplify(_star(d(_star(flat(v, g), g), coords), g).get((), 0))


def laplacian(f, coords, g):
    """Оператор Лапласа ⋆d⋆df на функциях."""
    return div(grad(f, coords, g), coords, g)


def physical(v, g):
    """Физические компоненты v̂ⁱ = √g_ii vⁱ (по ортонормированному базису) для диагональной метрики."""
    return [_no_abs(sp.sqrt(sp.Matrix(g)[i, i]) * c) for i, c in enumerate(v)]


def from_physical(v_hat, g):
    """Контравариантные компоненты vⁱ = v̂ⁱ/√g_ii для диагональной метрики."""
    return [_no_abs(c / sp.sqrt(sp.Matrix(g)[i, i])) for i, c in enumerate(v_hat)]
