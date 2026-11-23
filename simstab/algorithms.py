"""Model-free algorithms of the paper (no process noise, two-point zeroth-order estimation).

  Algorithm 1  two-point zeroth-order gradient estimate, averaged over the H systems
  Algorithm 2  simultaneously stabilizing all modes (discount annealing, full state)
  Algorithm 3  estimating the left unstable representation by adjoint (VJP) sampling
  Algorithm 4  simultaneously stabilizing the unstable modes on a shared subspace
  Algorithm 5  low-rank adaptation of a common controller to a downstream system

All of them are instances of one discounted policy-gradient loop over a parametrization
    K(theta) = K_base + theta Phi^T,  theta in R^{du x ell},
with stage cost x^T Qc x + (theta Phi^T x)^T R (theta Phi^T x) and rollouts on the physical
systems x_{t+1} = (A + B K(theta)) x_t:
  * Algorithm 2:  K_base = 0,   Phi = I,             Qc = Q
  * Algorithm 4:  K_base = 0,   Phi = shared Phi_hat, Qc = Pi Q Pi  (Pi = Phi_hat Phi_hat^T)
  * Algorithm 5:  K_base = K_c, Phi = Psi_hat,       Qc = Pi Q Pi  (eq. (96))
Exact model quantities are used only for logging (spectral radii, true gradients).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from . import lqr


# ---------------------------------------------------------------------------
# configuration and problem description
# ---------------------------------------------------------------------------

@dataclass
class PGConfig:
    eta: float = 1e-3        # step size
    N: int = 3               # inner policy-gradient updates per discount iteration
    n_s: int = 5             # two-point pairs per system per gradient estimate
    n_c: int = 20            # rollouts per system for the cost estimate
    r: float = 0.05          # smoothing radius (perturbation norm r * sqrt(d))
    tau: int = 100           # rollout horizon
    xi: float = 0.5          # discount increment factor
    gamma0: float = 0.2      # initial discount factor
    max_iters: int = 3000    # safeguard on the number of discount iterations
    log_grad_error: bool = True
    eps: float = 0.5         # target relative gradient error when n_s == "auto"

    def to_dict(self):
        return asdict(self)

    def samples_per_iter(self):
        """State transitions collected from EACH system in one discount iteration."""
        return (self.N * 2 * self.n_s + self.n_c) * self.tau


class Problem:
    """H physical systems and the controller parametrization K = K_base + theta Phi^T."""

    def __init__(self, A, B, Q, R, Phi=None, Kbase=None, project_cost=False):
        self.A, self.B, self.Q, self.R = A, B, Q, R
        self.H, self.dx, self.du = A.shape[0], A.shape[1], B.shape[2]
        Phi = np.eye(self.dx) if Phi is None else Phi
        self.Phi = Phi                                     # (dx, ell), shared
        self.ell = Phi.shape[1]
        self.Kbase = np.zeros((self.H, self.du, self.dx)) if Kbase is None else \
            np.broadcast_to(Kbase, (self.H, self.du, self.dx)).copy()
        self.kbase_zero = not np.any(self.Kbase)
        if project_cost:
            Pi = Phi @ Phi.T
            self.Qc = np.einsum("xy,hyz,zw->hxw", Pi, Q, Pi)
            self.q = min(np.linalg.eigvalsh(Phi.T @ Q[h] @ Phi).min() for h in range(self.H))
        else:
            self.Qc = Q
            self.q = min(np.linalg.eigvalsh(Q[h]).min() for h in range(self.H))
        self.d = self.du * self.ell

    def K(self, theta):
        """Physical controllers K_h = K_base,h + theta Phi^T, shape (H, du, dx)."""
        return self.Kbase + theta @ self.Phi.T


# ---------------------------------------------------------------------------
# rollouts and estimators
# ---------------------------------------------------------------------------

def sphere(rng, shape_lead, dim_shape, radius):
    """Uniform samples on the Frobenius sphere of the given radius."""
    U = rng.standard_normal(tuple(shape_lead) + tuple(dim_shape))
    axes = tuple(range(len(shape_lead), U.ndim))
    return radius * U / np.sqrt(np.sum(U ** 2, axis=axes, keepdims=True))


def initial_states(rng, H, M, dx):
    """Bounded isotropic resets: uniform on the sphere of radius sqrt(dx), so E[x0 x0^T] = I."""
    return sphere(rng, (H, M), (dx,), np.sqrt(dx))


def rollout_costs(prob: Problem, Theta, X0, gamma, tau):
    """Finite-horizon discounted costs V^{(h),gamma,tau}(theta, x0) without process noise.
    Theta: (H, M, du, ell), X0: (H, M, dx). Returns (H, M)."""
    # Per-system GEMMs: x (H, M, dx); the perturbed gains act through u = K x (du is small).
    Kth = Theta @ prob.Phi.T                                  # (H, M, du, dx)
    Ktot = Kth + prob.Kbase[:, None]
    AT = np.swapaxes(prob.A, 1, 2)                            # (H, dx, dx)
    BT = np.swapaxes(prob.B, 1, 2)                            # (H, du, dx)
    x = np.array(X0, dtype=float)
    cost = np.zeros(X0.shape[:2])
    disc = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(tau):
            v = np.einsum("hmux,hmx->hmu", Kth, x)               # input penalized in the cost
            u = v if prob.kbase_zero else np.einsum("hmux,hmx->hmu", Ktot, x)
            stage = np.einsum("hmx,hmx->hm", x @ prob.Qc, x) + np.einsum("hmu,hmu->hm", v @ prob.R, v)
            cost += disc * stage
            x = x @ AT + u @ BT
            disc *= gamma
    return cost


def zo2p_gradient(prob: Problem, theta, gamma, n_s, r, tau, rng):
    """Algorithm 1: two-point estimate, same reset for both rollouts of a pair, averaged over h."""
    H, d = prob.H, prob.d
    U = sphere(rng, (H, n_s), (prob.du, prob.ell), np.sqrt(d))
    X0 = initial_states(rng, H, n_s, prob.dx)
    Theta = np.concatenate([theta + r * U, theta - r * U], axis=1)
    V = rollout_costs(prob, Theta, np.concatenate([X0, X0], axis=1), gamma, tau)
    dV = V[:, :n_s] - V[:, n_s:]                                # (H, n_s)
    g_h = np.einsum("hi,hiuv->huv", dV, U) / (2 * r * n_s)
    return g_h.mean(axis=0)


def cost_estimates(prob: Problem, theta, gamma, n_c, tau, rng):
    """hat J^{(h),gamma,tau}(theta) from n_c rollouts per system, shape (H,)."""
    X0 = initial_states(rng, prob.H, n_c, prob.dx)
    Theta = np.broadcast_to(theta, (prob.H, n_c) + theta.shape)
    return rollout_costs(prob, Theta, X0, gamma, tau).mean(axis=1)


def accuracy_matched_ns(prob: Problem, gamma, eps, r, tau, rng, reps=200, theta=None, n_max=5000):
    """Smallest per-system batch n_s whose relative RMS gradient error at theta (default 0) is
    at most eps, i.e. n_s = ceil((e_1 / eps)^2) with e_1 the RMS error of a single pair per
    system. This is the batch rule of Lemma .9 / Corollaries .1 and .4 (H n_s ~ d / eps^2).
    Uses the exact gradient, so it is an experiment-design step, not part of the algorithm."""
    theta = np.zeros((prob.du, prob.ell)) if theta is None else theta
    g = lqr.exact_avg_grad(prob.A, prob.B, prob.Q, prob.R, theta, prob.Phi, gamma, prob.Kbase, prob.Qc)
    errs = [np.sum((zo2p_gradient(prob, theta, gamma, 1, r, tau, rng) - g) ** 2) for _ in range(reps)]
    e1 = np.sqrt(np.mean(errs)) / np.linalg.norm(g)
    return int(min(n_max, max(1, np.ceil((e1 / eps) ** 2)))), float(e1)


# ---------------------------------------------------------------------------
# discounted policy gradient (Algorithms 2, 4 and the inner loop of 5)
# ---------------------------------------------------------------------------

def discount_method(prob: Problem, cfg: PGConfig, rng, theta0=None, A_eval=None, B_eval=None,
                    offset_samples=0, verbose=False, tag=""):
    """Runs the discount annealing loop until gamma = 1.

    Logs, at the start of every discount iteration j and once at termination:
      gamma_j, cumulative samples per system, rho(A_1 + B_1 K_j), max_h rho(A_h + B_h K_j)
      over the training systems (and over A_eval/B_eval if given), and the relative error of an
      independent gradient estimate with respect to the exact averaged gradient (not counted
      in the samples).
    """
    theta = np.zeros((prob.du, prob.ell)) if theta0 is None else theta0.copy()
    gamma = float(cfg.gamma0)
    e1 = np.nan
    if cfg.n_s == "auto":
        cfg = PGConfig(**cfg.to_dict())
        cfg.n_s, e1 = accuracy_matched_ns(prob, gamma, cfg.eps, cfg.r, cfg.tau, rng, theta=theta)
        if verbose:
            print(f"[{tag}] accuracy-matched n_s = {cfg.n_s} (single-pair RMS error {e1:.3f})", flush=True)
    samples = float(offset_samples)
    per_iter = cfg.samples_per_iter()
    log = dict(gamma=[], samples=[], rho1=[], rho_max=[], grad_err=[], grad_err_abs=[],
               grad_norm=[], rho_eval_max=[], alpha_min=[])
    status = "max_iters"

    def record(theta, gamma, grad_err, grad_abs=np.nan, grad_norm=np.nan):
        K = prob.K(theta)
        radii = np.array([lqr.spectral_radius(prob.A[h] + prob.B[h] @ K[h]) for h in range(prob.H)])
        log["gamma"].append(gamma)
        log["samples"].append(samples)
        log["rho1"].append(radii[0])
        log["rho_max"].append(radii.max())
        log["grad_err"].append(grad_err)
        log["grad_err_abs"].append(grad_abs)
        log["grad_norm"].append(grad_norm)
        if A_eval is not None:
            Ke = prob.Kbase[0] + theta @ prob.Phi.T
            log["rho_eval_max"].append(lqr.closed_loop_radii(A_eval, B_eval, Ke).max())

    for j in range(cfg.max_iters):
        grad_err = grad_abs = grad_norm = np.nan
        if cfg.log_grad_error:
            g_true = lqr.exact_avg_grad(prob.A, prob.B, prob.Q, prob.R, theta, prob.Phi, gamma,
                                        prob.Kbase, prob.Qc)
            if g_true is not None:
                g_hat = zo2p_gradient(prob, theta, gamma, cfg.n_s, cfg.r, cfg.tau, rng)
                grad_abs = np.linalg.norm(g_hat - g_true)
                grad_norm = np.linalg.norm(g_true)
                grad_err = grad_abs / max(grad_norm, 1e-12)
        record(theta, gamma, grad_err, grad_abs, grad_norm)

        for _ in range(cfg.N):
            g = zo2p_gradient(prob, theta, gamma, cfg.n_s, cfg.r, cfg.tau, rng)
            theta = theta - cfg.eta * g
        J_hat = cost_estimates(prob, theta, gamma, cfg.n_c, cfg.tau, rng)
        samples += per_iter
        if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(J_hat))):
            status = "diverged"
            break
        alpha = prob.q / np.maximum(prob.q, 2 * J_hat - prob.q)
        log["alpha_min"].append(float(alpha.min()))
        gamma = min(1.0, (1 + cfg.xi * alpha.min()) * gamma)
        if verbose and j % 50 == 0:
            print(f"[{tag}] iter {j:5d}  gamma {gamma:.4f}  rho1 {log['rho1'][-1]:.4f}  "
                  f"rho_max {log['rho_max'][-1]:.4f}  min alpha {alpha.min():.2e}", flush=True)
        if gamma >= 1.0:
            status = "done"
            break

    record(theta, gamma, np.nan)
    out = {k: np.asarray(v, dtype=float) for k, v in log.items()}
    out["theta"] = theta
    out["K"] = prob.K(theta)[0] if prob.H == 1 else (theta @ prob.Phi.T + prob.Kbase[0])
    out["iterations"] = len(out["gamma"]) - 1
    out["status"] = status
    out["stabilized"] = bool(status == "done" and out["rho_max"][-1] < 1.0)
    out["total_samples"] = samples
    out["n_s"] = cfg.n_s
    out["e1"] = e1
    out["eta"] = cfg.eta
    return out


# ---------------------------------------------------------------------------
# Algorithm 3: left unstable representation by adjoint sampling
# ---------------------------------------------------------------------------

def make_vjp(Acl, backend="numpy"):
    """VJP of the simulator x -> f(x, 0) = Acl x, i.e. v -> grad_x (v^T f(x, 0)).
    backend="torch" differentiates a torch simulator with autograd (no Jacobian is formed);
    backend="numpy" uses the analytic adjoint Acl^T v, identical for linear simulators."""
    if backend == "torch":
        import torch
        At = torch.as_tensor(Acl, dtype=torch.float64)

        def vjp(V):
            out = []
            for v in V.T:
                x = torch.zeros(At.shape[0], dtype=torch.float64, requires_grad=True)
                y = At @ x
                (gx,) = torch.autograd.grad(torch.dot(torch.as_tensor(v), y), x)
                out.append(gx.numpy())
            return np.stack(out, axis=1)
        return vjp
    return lambda V: Acl.T @ V


def adjoint_representation(vjp, dx, ell, m, T, rng):
    """Algorithm 3: m restarts from a bounded isotropic distribution, T adjoint steps,
    top-ell left singular vectors. A common scalar rescaling per step avoids overflow and
    leaves the singular vectors unchanged."""
    V = sphere(rng, (m,), (dx,), np.sqrt(dx)).T                 # (dx, m)
    for _ in range(T):
        V = vjp(V)
        V = V / max(np.abs(V).max(), 1e-300)
    U, _, _ = np.linalg.svd(V, full_matrices=False)
    return U[:, :ell]


def shared_representation(A, ell, m, T, rng, backend="numpy"):
    """Algorithm 4, lines 1-5: per-system representations, averaged projector, top-ell eigvecs."""
    H, dx = A.shape[0], A.shape[1]
    Phis = [adjoint_representation(make_vjp(A[h], backend), dx, ell, m, T, rng) for h in range(H)]
    Pi = sum(P @ P.T for P in Phis) / H
    return lqr.top_eigvecs(Pi, ell), Phis


def true_shared_representation(A, ell):
    Phis = [lqr.left_unstable_basis(A[h]) for h in range(A.shape[0])]
    Pi = sum(P @ P.T for P in Phis) / len(Phis)
    return lqr.top_eigvecs(Pi, ell), Phis


# ---------------------------------------------------------------------------
# Algorithm 5: low-rank adaptation of a common controller
# ---------------------------------------------------------------------------

def adapt_low_rank(A, B, Q, R, Kc, ell_bar, cfg: PGConfig, m, T, rng, backend="numpy",
                   verbose=False, tag=""):
    """A, B, Q, R of ONE downstream system (no leading H axis). Returns the discount-method log
    with K_h = K_c + theta_h Psi_hat^T; samples include no adjoint calls (counted separately)."""
    if ell_bar == 0:
        return None
    Ac = A + B @ Kc
    Psi = adjoint_representation(make_vjp(Ac, backend), A.shape[0], ell_bar, m, T, rng)
    prob = Problem(A[None], B[None], Q[None], R[None], Phi=Psi, Kbase=Kc[None], project_cost=True)
    out = discount_method(prob, cfg, rng, verbose=verbose, tag=tag)
    out["Psi"] = Psi
    out["subspace_err"] = lqr.subspace_distance(Psi, lqr.left_unstable_basis(Ac))
    out["adjoint_calls"] = m * T
    return out
