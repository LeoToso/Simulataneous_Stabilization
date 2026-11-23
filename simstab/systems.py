"""Families of LTI systems used in the experiments.

Every family samples discrete-time pairs (A, B) together with cost matrices (Q, R).
`regime="train"` draws the systems used to learn the common controller and
`regime="downstream"` draws systems from a shifted parameter range, used to test the
low-rank adaptation of that controller.

Control convention (as in the paper): u_t = K x_t, closed loop A + B K.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np
import scipy.linalg as sla


def c2d_zoh(Ac: np.ndarray, Bc: np.ndarray, dt: float):
    """Zero-order-hold discretization."""
    n, m = Bc.shape
    E = sla.expm(np.block([[Ac, Bc], [np.zeros((m, n + m))]]) * dt)
    return E[:n, :n], E[:n, n:]


@dataclass
class Family:
    name: str
    dx: int
    du: int
    ell: int                      # number of unstable modes of every training system
    sampler: Callable             # sampler(rng, regime) -> (A, B)
    Q: np.ndarray
    R: np.ndarray
    description: str = ""
    extra: Dict = field(default_factory=dict)

    def sample(self, rng: np.random.Generator, n: int, regime: str = "train"):
        As, Bs = [], []
        for _ in range(n):
            A, B = self.sampler(rng, regime)
            As.append(A)
            Bs.append(B)
        A = np.stack(As)
        B = np.stack(Bs)
        Q = np.broadcast_to(self.Q, (n,) + self.Q.shape).copy()
        R = np.broadcast_to(self.R, (n,) + self.R.shape).copy()
        return A, B, Q, R


# ---------------------------------------------------------------------------
# Inverted pendulum (Euler, dt = 0.25), as in the original notebook
# ---------------------------------------------------------------------------

def _pendulum(rng, regime, dt=0.25, g=9.81):
    if regime == "train":
        m = 0.5 + rng.uniform(0.05, 0.2)
        ell = 0.75 + rng.uniform(0.05, 0.2)
    else:  # shorter and heavier pendulums: faster unstable mode, less authority
        m = 0.5 + rng.uniform(0.6, 1.0)
        ell = 0.45 + rng.uniform(0.0, 0.1)
    A = np.array([[1.0, dt], [(g / ell) * dt, 1.0]])
    B = np.array([[0.0], [dt / (m * ell ** 2)]])
    return A, B


# ---------------------------------------------------------------------------
# Cart-pole (Euler, dt = 0.25), as in the original notebook
# ---------------------------------------------------------------------------

def _cartpole(rng, regime, dt=0.25, g=9.8):
    if regime == "train":
        mc = 1.0 + 0.1 * rng.random()
        mp = 0.5 + 0.1 * rng.random()
        ell = 0.75 + 0.1 * rng.random()
    else:  # heavier pole on a lighter cart, shorter pole
        mc = 0.7 + 0.1 * rng.random()
        mp = 0.9 + 0.2 * rng.random()
        ell = 0.45 + 0.1 * rng.random()
    Ac = np.array([[0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, (mp / mc) * g, 0.0],
                   [0.0, 0.0, 0.0, 1.0],
                   [0.0, 0.0, g * (mp + mc) / (ell * mc), 0.0]])
    Bc = np.array([[0.0], [1.0 / mc], [0.0], [1.0 / (ell * mc)]])
    return np.eye(4) + dt * Ac, dt * Bc


# ---------------------------------------------------------------------------
# Launch vehicle pitch plane (Wie, Du, Whorton, AIAA 2008-6291), ZOH, dt = 0.25
# x = [theta, theta_dot, alpha], u = gimbal deflection
# ---------------------------------------------------------------------------

_LV = dict(Iy=2.43e6, m=5830.0, T=341_000.0, F=375_000.0, x_cp=38.0, x_cg=32.3,
           V=1320.0, N_alpha=240_000.0)


def _launch(rng, regime, dt=0.25):
    u = rng.uniform
    if regime == "train":
        m = _LV["m"] * u(0.9, 1.5)
        Iy = _LV["Iy"] * u(0.9, 1.5)
        T = _LV["T"] * u(0.9, 1.0)
        F = _LV["F"] * u(0.9, 1.0)
        x_cp = _LV["x_cp"] * u(0.95, 1.0)
        x_cg = _LV["x_cg"] * u(0.95, 1.0)
        V = _LV["V"] * u(0.9, 1.0)
        N_alpha = _LV["N_alpha"] * u(0.95, 1.1)
    else:  # max-q like conditions: larger aerodynamic moment, less thrust
        m = _LV["m"] * u(0.9, 1.1)
        Iy = _LV["Iy"] * u(0.6, 0.8)
        T = _LV["T"] * u(0.6, 0.75)
        F = _LV["F"] * u(0.9, 1.0)
        x_cp = _LV["x_cp"] * u(1.05, 1.15)
        x_cg = _LV["x_cg"] * u(0.95, 1.0)
        V = _LV["V"] * u(0.9, 1.0)
        N_alpha = _LV["N_alpha"] * u(1.4, 1.8)
    M_alpha = x_cp * N_alpha / Iy
    M_delta = x_cg * T / Iy
    Ac = np.array([[0.0, 1.0, 0.0],
                   [0.0, 0.0, M_alpha],
                   [-F / (m * V), 1.0, -N_alpha / (m * V)]])
    Bc = np.array([[0.0], [M_delta], [T / (m * V)]])
    return c2d_zoh(Ac, Bc, dt)


# ---------------------------------------------------------------------------
# Flexible arm: planar Hencky bar chain (Lutkus, Lindemann, Tu, arXiv:2610.40931, Sec. 6.1)
#   M(q) q'' + C(q,q') q' + (c0 M + c1 K_eff) q' + kappa K0 q - S^T (w * sin(S q)) = 1 u
# linearized at the upright configuration q = 0 and discretized by ZOH (dt = 0.05).
# x = (q, q'), dx = 2 n_s, du = 1. kappa lies between the n_unstable-th and the
# (n_unstable+1)-th largest generalized eigenvalue of (S^T diag(w) S, K0), so exactly
# n_unstable bending modes buckle. Nominal parameters reproduce the paper's spectrum
# (unstable eigenvalue 1.129; slowest stable 0.874 and 0.581 +- 0.633j) and |B| = 2.14,
# which is normalized to 1 (input in units of 2.14^-1 N m).
# ---------------------------------------------------------------------------

def flexible_arm(n_sections=20, sec_len=0.05, sec_mass=0.04, g=9.81, c0=0.4, c1=0.02,
                 k_spread=(1.0, 1.5), n_unstable=1, kappa_frac=0.5, dt=0.05,
                 b_scale=None):
    ns = n_sections
    S = np.tril(np.ones((ns, ns)))
    J = np.tril(np.full((ns, ns), sec_len), -1) + np.eye(ns) * sec_len / 2
    M_phi = sec_mass * J.T @ J + sec_mass * sec_len ** 2 / 12 * np.eye(ns)  # uniform bars
    M = S.T @ M_phi @ S
    w = (ns - np.arange(1, ns + 1) + 0.5) * g * sec_len * sec_mass
    G = S.T @ np.diag(w) @ S
    K0 = np.diag(np.linspace(k_spread[0], k_spread[1], ns))
    ev = np.sort(sla.eigh(G, K0, eigvals_only=True))[::-1]
    kappa = (1 - kappa_frac) * ev[n_unstable - 1] + kappa_frac * ev[n_unstable]
    K_eff = kappa * K0 - G
    C_d = c0 * M + c1 * K_eff
    Mi = np.linalg.inv(M)
    Ac = np.block([[np.zeros((ns, ns)), np.eye(ns)], [-Mi @ K_eff, -Mi @ C_d]])
    Bc = np.vstack([np.zeros((ns, 1)), Mi @ np.ones((ns, 1))])
    A, B = c2d_zoh(Ac, Bc, dt)
    if b_scale is None:
        b_scale = np.linalg.norm(B, 2)
    return A, B / b_scale


_ARM_NOMINAL_B = {}


def _arm_sampler(n_unstable):
    def sampler(rng, regime):
        u = rng.uniform
        if n_unstable not in _ARM_NOMINAL_B:
            _, Bn = flexible_arm(n_unstable=n_unstable, b_scale=1.0)
            _ARM_NOMINAL_B[n_unstable] = np.linalg.norm(Bn, 2)
        if regime == "train":
            p = dict(sec_len=0.05 * u(0.97, 1.03), sec_mass=0.04 * u(0.9, 1.1),
                     c0=0.4 * u(0.9, 1.1), c1=0.02 * u(0.9, 1.1),
                     kappa_frac=u(0.4, 0.6))
        else:  # heavier, longer, less damped arm with softer springs
            p = dict(sec_len=0.05 * u(1.05, 1.10), sec_mass=0.04 * u(1.3, 1.5),
                     c0=0.4 * u(0.5, 0.7), c1=0.02 * u(0.5, 0.7),
                     kappa_frac=u(0.7, 0.85))
        return flexible_arm(n_unstable=n_unstable, b_scale=_ARM_NOMINAL_B[n_unstable], **p)
    return sampler


FAMILIES: Dict[str, Callable[[], Family]] = {
    "pendulum": lambda: Family(
        "pendulum", 2, 1, 1, _pendulum, np.eye(2), np.eye(1),
        "Inverted pendulum, Euler dt=0.25, random mass and length"),
    "cartpole": lambda: Family(
        "cartpole", 4, 1, 3, _cartpole, 100 * np.eye(4), 0.1 * np.eye(1),
        "Cart-pole, Euler dt=0.25, random cart mass, pole mass and length"),
    "launch": lambda: Family(
        "launch", 3, 1, 2, _launch, np.eye(3), np.eye(1),
        "Launch-vehicle pitch plane (Wie et al. 2008), ZOH dt=0.25"),
    "flexarm": lambda: Family(
        "flexarm", 40, 1, 1, _arm_sampler(1), np.eye(40), np.eye(1),
        "Flexible arm, 20-section bar chain, ZOH dt=0.05, 1 unstable mode"),
    "flexarm3": lambda: Family(
        "flexarm3", 40, 1, 3, _arm_sampler(3), np.eye(40), np.eye(1),
        "Flexible arm, 20-section bar chain, ZOH dt=0.05, 3 unstable modes"),
}


def get_family(name: str) -> Family:
    if name not in FAMILIES:
        raise KeyError(f"unknown system '{name}', choose from {sorted(FAMILIES)}")
    return FAMILIES[name]()
