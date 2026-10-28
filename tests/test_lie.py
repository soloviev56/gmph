import sympy as sp

from gmph.lie import adjoint, commutator, coordinates, killing_form, structure_constants

# вещественные генераторы вращений по конвенциям: (J_k)_ij = −ε_kij, [J_x, J_y] = J_z
J = [sp.Matrix(3, 3, lambda i, j: -sp.LeviCivita(k, i, j)) for k in range(3)]


def test_so3_structure_constants():
    f = structure_constants(J)
    for a in range(3):
        for b in range(3):
            assert f[a][b] == [sp.LeviCivita(a, b, c) for c in range(3)]
    assert commutator(J[0], J[1]) == J[2]


def test_adjoint_of_so3_is_itself_and_killing_negative():
    assert adjoint(J) == J
    assert killing_form(J) == -2 * sp.eye(3)


def test_sl2():
    H = sp.Matrix([[1, 0], [0, -1]])
    E = sp.Matrix([[0, 1], [0, 0]])
    F = sp.Matrix([[0, 0], [1, 0]])
    f = structure_constants([H, E, F])
    assert f[0][1] == [0, 2, 0] and f[0][2] == [0, 0, -2] and f[1][2] == [1, 0, 0]
    assert killing_form([H, E, F]) == sp.Matrix([[8, 0, 0], [0, 0, 4], [0, 4, 0]])
    assert coordinates(sp.Matrix([[2, 3], [5, -2]]), [H, E, F]) == [2, 3, 5]
