"""Experiments for one system family: everything for one seed runs in one worker process."""
from __future__ import annotations

import copy
import time

import numpy as np

from . import lqr
from .algorithms import (PGConfig, Problem, discount_method, shared_representation,
                         true_shared_representation, adapt_low_rank, zo2p_gradient)
from .systems import get_family


def _cfg(base: dict, override: dict, **extra) -> PGConfig:
    d = copy.deepcopy(base)
    d.pop("eps", None)
    d.update(override or {})
    d.update(extra)
    return PGConfig(**d)


def auto_gamma0(rho_bar, cap):
    """0 < gamma_0 < min{1, rho_bar^-2} (Algorithm 2 initialization), capped by `cap`."""
    return float(min(cap, 0.9 / rho_bar ** 2, 1.0))


def _strip(out):
    """Keep what the figures and the summary need."""
    keep = ("gamma", "samples", "rho1", "rho_max", "grad_err", "grad_err_abs", "grad_norm", "rho_eval_max", "alpha_min",
            "iterations", "status", "stabilized", "total_samples", "K", "subspace_err",
            "adjoint_calls", "n_s", "e1", "eta")
    return {k: out[k] for k in keep if k in out}


def run_seed(system: str, cfg: dict, seed: int, verbose: bool = False):
    """Main experiment and adaptation experiment for one seed."""
    fam = get_family(system)
    rng = np.random.default_rng(seed)
    H = cfg["H"]
    A, B, Q, R = fam.sample(rng, H, "train")
    At, Bt, _, _ = fam.sample(rng, cfg.get("n_test", 200), "train")        # held-out systems
    rho_bar = max(lqr.spectral_radius(a) for a in A)
    g0 = auto_gamma0(rho_bar, cfg["base"].get("gamma0", 1.0))
    res = dict(seed=seed, system=system, H=H)
    t0 = time.time()

    # --- Algorithm 2 on system 1 alone (single task, all modes) ---------------------------------
    c = _cfg(cfg["base"], cfg["single"], gamma0=g0, eps=cfg["eps"])
    out = discount_method(Problem(A[:1], B[:1], Q[:1], R[:1]), c, rng, verbose=verbose,
                          tag=f"{system}/single/s{seed}")
    res["single"] = _strip(out)

    # --- Algorithm 2 on the H systems (multitask, all modes) ------------------------------------
    c = _cfg(cfg["base"], cfg["multi"], gamma0=g0, eps=cfg["eps"])
    out = discount_method(Problem(A, B, Q, R), c, rng, verbose=verbose, tag=f"{system}/multi/s{seed}")
    res["multi"] = _strip(out)

    # --- Algorithm 4 on the H systems (shared unstable subspace) --------------------------------
    m, T = cfg["adj_restarts"], cfg["adj_horizon"]
    Phi_hat, _ = shared_representation(A, fam.ell, m, T, rng, cfg.get("autodiff", "numpy"))
    Phi_true, _ = true_shared_representation(A, fam.ell)
    c = _cfg(cfg["base"], cfg["subspace"], gamma0=g0, eps=cfg["eps"])
    out = discount_method(Problem(A, B, Q, R, Phi=Phi_hat, project_cost=True), c, rng,
                          verbose=verbose, tag=f"{system}/subspace/s{seed}")
    out["subspace_err"] = lqr.subspace_distance(Phi_hat, Phi_true)
    out["adjoint_calls"] = m * T
    res["subspace"] = _strip(out)

    for key in ("single", "multi", "subspace"):
        K = res[key]["K"]
        radii = lqr.closed_loop_radii(At, Bt, K)
        res[key]["heldout_frac"] = float(np.mean(radii < 1))
        res[key]["train_rho_max"] = float(lqr.closed_loop_radii(A, B, K).max())

    # --- Algorithm 5: adapting the common controller to downstream systems ----------------------
    ad = cfg.get("adapt")
    if ad:
        src = "subspace" if res["subspace"]["stabilized"] else "multi"
        Kc = res[src]["K"]
        res["adapt"] = run_adaptation(fam, cfg, Kc, rng, verbose, seed)
        res["adapt"]["source"] = src
    res["time"] = time.time() - t0
    return res


def run_adaptation(fam, cfg, Kc, rng, verbose=False, seed=0):
    """Downstream systems that the common controller K_c does NOT stabilize.
    For each one: low-rank adaptation (Algorithm 5), full-rank fine-tuning from K_c
    (Algorithm 2 with K_base = K_c, Phi = I) and learning from scratch (Algorithm 2, K_0 = 0)."""
    ad = cfg["adapt"]
    systems, tries = [], 0
    while len(systems) < ad["n_down"] and tries < 200 * ad["n_down"]:
        A, B, Q, R = fam.sample(rng, 1, "downstream")
        tries += 1
        if lqr.spectral_radius(A[0] + B[0] @ Kc) >= 1:
            systems.append((A[0], B[0], Q[0], R[0]))
    out = dict(n_found=len(systems), tries=tries, lora=[], finetune=[], scratch=[], ell_bar=[],
               rho_c=[])
    m, T = cfg["adj_restarts"], ad.get("adj_horizon", cfg["adj_horizon"])
    for i, (A, B, Q, R) in enumerate(systems):
        Ac = A + B @ Kc
        rho_c = lqr.spectral_radius(Ac)
        ell_bar = lqr.num_unstable(Ac)          # the rank is supplied to Algorithm 5
        out["ell_bar"].append(ell_bar)
        out["rho_c"].append(rho_c)
        g0c = auto_gamma0(rho_c, 1.0)
        c = _cfg(cfg["base"], ad.get("lora"), gamma0=g0c, eps=cfg["eps"])
        o = adapt_low_rank(A, B, Q, R, Kc, ell_bar, c, max(m, ell_bar), T, rng,
                           cfg.get("autodiff", "numpy"), verbose, f"lora/s{seed}/{i}")
        out["lora"].append(_strip(o))
        c = _cfg(cfg["base"], ad.get("finetune"), gamma0=g0c, eps=cfg["eps"])
        o = discount_method(Problem(A[None], B[None], Q[None], R[None], Kbase=Kc[None]), c, rng,
                            verbose=verbose, tag=f"finetune/s{seed}/{i}")
        out["finetune"].append(_strip(o))
        c = _cfg(cfg["base"], ad.get("scratch"), gamma0=auto_gamma0(lqr.spectral_radius(A), 1.0),
                 eps=cfg["eps"])
        o = discount_method(Problem(A[None], B[None], Q[None], R[None]), c, rng,
                            verbose=verbose, tag=f"scratch/s{seed}/{i}")
        out["scratch"].append(_strip(o))
    return out


# ---------------------------------------------------------------------------
# side experiments (cheap, model-based checks of Lemma .9 and Theorem .3)
# ---------------------------------------------------------------------------

def grad_error_vs_H(system: str, cfg: dict, seed: int = 0, Hs=None, reps: int = 30):
    """Relative error of the averaged two-point estimate at K = 0, gamma_0, with n_s pairs per
    system, for the full-state (d = du dx) and the shared-subspace (d = du ell) parametrizations."""
    fam = get_family(system)
    rng = np.random.default_rng(10_000 + seed)
    Hmax = cfg["H"]
    Hs = Hs or sorted({1, 2, 5, 10, 20, Hmax} & set(range(1, Hmax + 1)))
    A, B, Q, R = fam.sample(rng, Hmax, "train")
    g0 = auto_gamma0(max(lqr.spectral_radius(a) for a in A), cfg["base"].get("gamma0", 1.0))
    Phi, _ = true_shared_representation(A, fam.ell)
    base = cfg["base"]
    out = dict(H=list(Hs), n_s=1)
    for key in ("full", "subspace"):
        for suf in ("", "_std", "_q25", "_q75"):
            out[key + suf] = []
    for H in Hs:
        for key, proj in (("full", False), ("subspace", True)):
            p = Problem(A[:H], B[:H], Q[:H], R[:H], Phi=Phi if proj else None, project_cost=proj)
            th = np.zeros((p.du, p.ell))
            gt = lqr.exact_avg_grad(p.A, p.B, p.Q, p.R, th, p.Phi, g0, p.Kbase, p.Qc)
            e = [np.linalg.norm(zo2p_gradient(p, th, g0, 1, base["r"], base["tau"], rng) - gt)
                 / np.linalg.norm(gt) for _ in range(reps)]
            out[key].append(float(np.mean(e)))
            out[key + "_std"].append(float(np.std(e)))
            out[key + "_q25"].append(float(np.percentile(e, 25)))
            out[key + "_q75"].append(float(np.percentile(e, 75)))
    return out


def subspace_vs_T(system: str, cfg: dict, seeds=range(5), Ts=None):
    """Distance between the estimated and the true shared unstable subspace vs adjoint horizon T."""
    fam = get_family(system)
    Ts = Ts or list(range(1, 41))
    D = np.zeros((len(list(seeds)), len(Ts)))
    for si, s in enumerate(seeds):
        rng = np.random.default_rng(20_000 + s)
        A, _, _, _ = fam.sample(rng, cfg["H"], "train")
        Phi_true, _ = true_shared_representation(A, fam.ell)
        for ti, T in enumerate(Ts):
            Phi_hat, _ = shared_representation(A, fam.ell, cfg["adj_restarts"], T, rng)
            D[si, ti] = lqr.subspace_distance(Phi_hat, Phi_true)
    return dict(T=Ts, dist=D)
