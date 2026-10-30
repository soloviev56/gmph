"""Конечные группы и их представления (статья «Представления конечных групп»).

Группа задаётся образующими в точном матричном представлении; элементы
нумеруются кратчайшими словами в образующих. Любое другое представление
задаётся образами тех же образующих и вычисляется по словам.
"""

import sympy as sp


class FiniteGroup:
    """Конечная группа, порождённая матрицами ``generators = {имя: матрица}``.

    ``words[k]`` — кратчайшее слово k-го элемента (``""`` — единица, она первая),
    ``elements[k]`` — его матрица, ``classes`` — классы сопряжённости (списки номеров).
    """

    def __init__(self, generators):
        self.generators = dict(generators)
        identity = sp.eye(next(iter(self.generators.values())).rows)
        index = {tuple(identity): 0}
        self.words, self.elements = [""], [identity]
        frontier = [0]
        while frontier:
            new = []
            for k in frontier:
                for name, g in self.generators.items():
                    product = self.elements[k] * g
                    if tuple(product) not in index:
                        index[tuple(product)] = len(self.elements)
                        new.append(len(self.elements))
                        self.words.append(self.words[k] + name)
                        self.elements.append(product)
            frontier = new
        self._index = index
        self.classes = self._conjugacy_classes()

    @property
    def order(self):
        return len(self.elements)

    def index_of(self, matrix):
        return self._index[tuple(matrix)]

    def _conjugacy_classes(self):
        seen, classes = set(), []
        for k, a in enumerate(self.elements):
            if k in seen:
                continue
            cls = sorted({self.index_of(g * a * g.inv()) for g in self.elements})
            seen.update(cls)
            classes.append(cls)
        return classes

    def represent(self, images):
        """Матрицы представления, в котором образующие переходят в ``images``."""
        size = next(iter(images.values())).rows
        mats = []
        for word in self.words:
            m = sp.eye(size)
            for name in word:
                m = m * images[name]
            mats.append(m)
        return mats


def character(mats):
    """Характер χ(g) = tr ρ(g) для списка матриц представления."""
    return [sp.nsimplify(sp.simplify(m.trace())) for m in mats]


def inner(chi, psi):
    """Скалярное произведение характеров ⟨χ, ψ⟩ = (1/|G|) Σ_g conj(χ(g)) ψ(g)."""
    return sp.simplify(sum(sp.conjugate(a) * b for a, b in zip(chi, psi)) / len(chi))


def projector(mats, chi):
    """Проектор на изотипическую компоненту неприводимого характера χ:
    P = (d/|G|) Σ_g conj(χ(g)) ρ(g), d = χ(e)."""
    total = sp.zeros(mats[0].rows)
    for c, m in zip(chi, mats):
        total += sp.conjugate(c) * m
    return (chi[0] * total / len(mats)).applyfunc(sp.simplify)


def commutant_dimension(mats):
    """Размерность пространства матриц X, коммутирующих со всеми ρ(g)."""
    n = mats[0].rows
    X = sp.Matrix(n, n, sp.symbols(f"x0:{n * n}"))
    equations = []
    for m in mats:
        equations.extend(m * X - X * m)
    A = sp.Matrix([[sp.diff(e, v) for v in X] for e in equations])
    return n * n - A.rank()
