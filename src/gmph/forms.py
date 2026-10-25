"""Работа с дифференциальными формами поверх ``sympy.diffgeom``.

``sympy.diffgeom`` хранит координаты как скалярные поля (``BaseScalarField``),
а не как обычные символы. Здесь собраны тонкие обёртки, которые убирают
повторяющийся код в статьях и возвращают результаты в обычных символах SymPy.
"""

from dataclasses import dataclass

import sympy as sp
from sympy.diffgeom import BaseScalarField, BaseVectorField, CoordSystem, Manifold, Patch


@dataclass(frozen=True)
class Chart:
    """Карта (система координат) вместе с её базисами.

    Атрибуты:
        system:  объект ``CoordSystem``;
        symbols: координаты как обычные символы SymPy;
        coords:  координаты как скалярные поля на многообразии;
        vectors: базисные векторы ∂/∂x^i;
        forms:   базисные 1-формы dx^i.
    """

    system: CoordSystem
    symbols: tuple
    coords: tuple
    vectors: tuple
    forms: tuple


def chart(names, name="C", relations=None, **assumptions):
    """Создать карту на новом многообразии размерности ``len(names)``.

    ``names`` — строка вида ``"x y z"`` или список строк. Дополнительные
    именованные аргументы (например, ``real=True``) передаются в ``sp.symbols``.

    >>> X = chart("x y", real=True)
    >>> dx, dy = X.forms
    """
    symbols = tuple(sp.symbols(names, **assumptions))
    patch = Patch("P", Manifold("M", len(symbols)))
    return _make_chart(CoordSystem(name, patch, symbols, relations=relations or {}))


def _make_chart(system):
    return Chart(
        system=system,
        symbols=tuple(system.symbols),
        coords=tuple(system.coord_functions()),
        vectors=tuple(system.base_vectors()),
        forms=tuple(system.base_oneforms()),
    )


def to_symbols(expr, chart_):
    """Выразить скалярное выражение в координатах карты ``chart_``.

    Координатные поля самой карты заменяются её символами, а поля других
    карт — их выражениями через координаты ``chart_`` (по заданным связям).
    """
    expr = sp.sympify(expr)
    subs = {}
    for field in expr.atoms(BaseScalarField):
        system = field._coord_sys
        if system == chart_.system:
            subs[field] = chart_.symbols[field._index]
        else:
            subs[field] = chart_.system.transform(system)[field._index]
    return expr.subs(subs)


def pair(form, *vectors, chart=None):
    """Значение формы на векторах, упрощённое и выраженное в символах.

    Результат записывается в координатах ``chart``; по умолчанию — в той
    карте, в которой заданы векторы (она должна быть одна).
    """
    if chart is None:
        chart = _chart_of(vectors)
    return sp.simplify(to_symbols(form.rcall(*vectors), chart))


def components(form, chart_):
    """Компоненты 1-формы ω_i = ω(∂/∂x^i) в данной карте."""
    return [pair(form, e, chart=chart_) for e in chart_.vectors]


def matrix(tensor, chart_):
    """Матрица компонент T_ij = T(∂_i, ∂_j) ковариантного 2-тензора в данной карте."""
    e = chart_.vectors
    return sp.Matrix(len(e), len(e), lambda i, j: pair(tensor, e[i], e[j], chart=chart_))


def _chart_of(vectors):
    systems = {f._coord_sys for v in vectors for f in v.atoms(BaseVectorField)}
    if len(systems) != 1:
        raise ValueError("векторы заданы в разных картах: укажите chart явно")
    return _make_chart(systems.pop())


def plane_polar():
    """Плоскость с двумя связанными картами: декартовой (x, y) и полярной (r, φ).

    Возвращает пару ``(cart, polar)``. Полярная карта покрывает плоскость
    без луча φ = π и начала координат; r > 0.
    """
    x, y = sp.symbols("x y", real=True)
    r, phi = sp.symbols("r phi", positive=True)
    relations = {
        ("polar", "cart"): [(r, phi), (r * sp.cos(phi), r * sp.sin(phi))],
        ("cart", "polar"): [(x, y), (sp.sqrt(x**2 + y**2), sp.atan2(y, x))],
    }
    patch = Patch("P", Manifold("M", 2))
    cart = CoordSystem("cart", patch, [x, y], relations=relations)
    polar = CoordSystem("polar", patch, [r, phi], relations=relations)
    return _make_chart(cart), _make_chart(polar)
