"""Exact (model-based) quantities. They are used only for evaluation and logging
(spectral radii, true gradients, ground-truth subspaces), never by the learning algorithms."""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

UNIT_TOL = 1e-9


def spectral_radius(M: np.ndarray) -> float:
    if not np.all(np.isfinite(M)):
        return np.inf
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def closed_loop_radii(A, B, K):
    """rho(A_h + B_h K) for every h. K is (du, dx) or (H, du, dx)."""
    K = np.broadcast_to(K, (A.shape[0],) + K.shape[-2:])
    return np.array([spectral_radius(A[h] + B[h] @ K[h]) for h in range(A.shape[0])])


def num_unstable(A, tol=UNIT_TOL):
    return int(np.sum(np.abs(np.linalg.eigvals(A)) >= 1 - tol))


def left_unstable_basis(A, tol=UNIT_TOL):
    """Orthonormal basis of the left unstable invariant subspace of A
    (invariant subspace of A^T for eigenvalues with |lambda| >= 1), via an ordered real Schur form."""
    T, Z, sdim = sla.schur(A.T, output="real", sort=lambda re, im: re * re + im * im >= (1 - tol) ** 2)
    return Z[:, :sdim]


def subspace_distance(P1, P2):
    return float(np.linalg.norm(P1 @ P1.T - P2 @ P2.T, 2))


def top_eigvecs(S, ell):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return V[:, np.argsort(w)[::-1][:ell]]


def dlyap_value(F, W):
    """P = W + F^T P F."""
    return sla.solve_discrete_lyapunov(F.T, W)


def exact_cost_grad(A, B, Q, R, theta, Phi, gamma, Kbase=None, Qc=None):
    """Discounted cost J = Tr(P) (E x0 x0^T = I) of the controller K = Kbase + theta Phi^T with
    stage cost x^T Qc x + (theta Phi^T x)^T R (theta Phi^T x), and its gradient w.r.t. theta.
    Returns (inf, None) if sqrt(gamma)(A + B K) is not Schur stable."""
    dx = A.shape[0]
    Kth = theta @ Phi.T
    K = Kth if Kbase is None else Kbase + Kth
    Acl = A + B @ K
    F = np.sqrt(gamma) * Acl
    if spectral_radius(F) >= 1:
        return np.inf, None
    Qc = Q if Qc is None else Qc
    P = dlyap_value(F, Qc + Kth.T @ R @ Kth)
    Sig = sla.solve_discrete_lyapunov(F, np.eye(dx))
    grad = 2 * (R @ Kth + gamma * B.T @ P @ Acl) @ Sig @ Phi
    return float(np.trace(P)), grad


def exact_avg_grad(A, B, Q, R, theta, Phi, gamma, Kbase=None, Qc=None):
    H = A.shape[0]
    g = 0.0
    for h in range(H):
        Kb = None if Kbase is None else Kbase[h]
        Qch = None if Qc is None else Qc[h]
        _, gh = exact_cost_grad(A[h], B[h], Q[h], R[h], theta, Phi, gamma, Kb, Qch)
        if gh is None:
            return None
        g = g + gh / H
    return g
