import numpy as np

from gmph.integrators import integrate, leapfrog_step, rk4_step, yoshida4_step


def pendulum_force(q):
    return -np.sin(q)


def step_jacobian(step, q, p, h, eps=1e-6):
    """Матрица Якоби одного шага по (q, p) центральными разностями (одна степень свободы)."""
    cols = []
    for dq, dp in [(eps, 0.0), (0.0, eps)]:
        plus = np.array(step(pendulum_force, q + dq, p + dp, h))
        minus = np.array(step(pendulum_force, q - dq, p - dp, h))
        cols.append((plus - minus) / (2 * eps))
    return np.array(cols).T


def test_symplectic_steps_preserve_area():
    for step, symplectic in [(leapfrog_step, True), (yoshida4_step, True), (rk4_step, False)]:
        det = np.linalg.det(step_jacobian(step, 1.0, 0.5, 0.5))
        assert (abs(det - 1.0) < 1e-9) == symplectic


def test_orders_of_convergence():
    def error(step, h):
        n = int(round(2.0 / h))
        q, p = integrate(step, pendulum_force, 1.0, 0.0, h, n)
        q_ref, p_ref = integrate(yoshida4_step, pendulum_force, 1.0, 0.0, h / 16, 16 * n)
        return abs(q[-1] - q_ref[-1]) + abs(p[-1] - p_ref[-1])
    for step, order in [(leapfrog_step, 2), (yoshida4_step, 4), (rk4_step, 4)]:
        ratio = error(step, 0.1) / error(step, 0.05)
        assert abs(np.log2(ratio) - order) < 0.3


def test_leapfrog_is_time_reversible():
    q1, p1 = leapfrog_step(pendulum_force, 0.7, 0.2, 0.25)
    q0, p0 = leapfrog_step(pendulum_force, q1, -p1, 0.25)
    assert np.allclose([q0, -p0], [0.7, 0.2])
