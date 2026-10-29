"""Интеграторы с постоянным шагом для гамильтоновых систем H = p²/2m + V(q).

Состояние — пара массивов (q, p) одинаковой формы; ``force(q) = −∇V(q)``.
Шаг ``step(force, q, p, h, mass)`` возвращает новую пару (q, p). Схема
Штёрмера–Верле (leapfrog) и композиция Йошиды симплектичны; RK4 дан для
сравнения и симплектическим не является.
"""

import numpy as np

# коэффициенты «тройного прыжка» Йошиды: w₁ + w₀ + w₁ = 1, w₁³ + w₀³ + w₁³ = 0
_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_W0 = 1.0 - 2.0 * _W1


def leapfrog_step(force, q, p, h, mass=1.0):
    """Шаг Штёрмера–Верле: полтолчка — дрейф — полтолчка (второй порядок)."""
    p_half = p + 0.5 * h * force(q)
    q_new = q + h * p_half / mass
    return q_new, p_half + 0.5 * h * force(q_new)


def yoshida4_step(force, q, p, h, mass=1.0):
    """Симплектический шаг четвёртого порядка: три шага Верле с шагами w₁h, w₀h, w₁h."""
    for w in (_W1, _W0, _W1):
        q, p = leapfrog_step(force, q, p, w * h, mass)
    return q, p


def rk4_step(force, q, p, h, mass=1.0):
    """Классический метод Рунге–Кутты 4-го порядка для q̇ = p/m, ṗ = F(q)."""
    def f(qq, pp):
        return pp / mass, force(qq)
    k1 = f(q, p)
    k2 = f(q + 0.5 * h * k1[0], p + 0.5 * h * k1[1])
    k3 = f(q + 0.5 * h * k2[0], p + 0.5 * h * k2[1])
    k4 = f(q + h * k3[0], p + h * k3[1])
    q_new = q + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    p_new = p + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return q_new, p_new


def integrate(step, force, q0, p0, h, n, mass=1.0):
    """n шагов метода ``step``; возвращает массивы q, p формы (n + 1, …)."""
    q, p = np.asarray(q0, dtype=float), np.asarray(p0, dtype=float)
    qs, ps = [q], [p]
    for _ in range(n):
        q, p = step(force, q, p, h, mass)
        qs.append(q)
        ps.append(p)
    return np.array(qs), np.array(ps)
