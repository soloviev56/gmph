import sympy as sp

from gmph.finite_groups import FiniteGroup, character, commutant_dimension, inner, projector


def _s3():
    t = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    c = sp.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    return FiniteGroup({"t": t, "c": c})


def test_s3_order_and_classes():
    G = _s3()
    assert G.order == 6
    assert sorted(len(cls) for cls in G.classes) == [1, 2, 3]


def test_character_orthogonality_and_decomposition():
    G = _s3()
    th = 2 * sp.pi / 3
    E = G.represent({"t": sp.Matrix([[1, 0], [0, -1]]),
                     "c": sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])})
    sign = G.represent({"t": sp.Matrix([[-1]]), "c": sp.Matrix([[1]])})
    chi_E, chi_sign, chi_perm = character(E), character(sign), character(G.elements)
    assert inner(chi_E, chi_E) == 1 and inner(chi_E, chi_sign) == 0
    assert inner(chi_E, chi_perm) == 1 and inner(chi_sign, chi_perm) == 0
    P = projector(G.elements, chi_E)
    assert P * P == P and P.rank() == 2
    assert commutant_dimension(E) == 1 and commutant_dimension(G.elements) == 2
