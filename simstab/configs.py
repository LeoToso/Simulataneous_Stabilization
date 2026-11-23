"""Hyperparameters of every system.

Protocol (the same for all systems):
  * All settings share N, n_c, r, tau, xi and gamma_0 = 0.9 / max_h rho(A_h)^2.
  * Adaptation methods (one downstream system each) share a larger n_c (`nc_adapt`): with a
    single system there is no min over h in the discount update, and a cost underestimate from
    few resets can end the schedule early (Lemma .24 needs n_c ~ mu_0^4, mu_0^2 = d_X).
  * n_s = "auto": per-system two-point batch matched to a common gradient accuracy `eps`
    (relative RMS error at initialization), n_s = ceil((e_1 / eps)^2), the batch rule of
    Lemma .9 / Corollaries .1 and .4 (H n_s ~ d / eps^2). Collaboration (larger H) and the
    shared subspace (d = du ell instead of du dx) therefore show up as smaller per-system
    batches. An integer n_s fixes the batch instead.
  * eta is per setting (the Lipschitz constant, hence the admissible step, depends on the
    parametrization). Values come from `python tune.py`: one grid step below the largest step
    that stabilized every tuning seed (the largest one is at the edge of divergence near
    gamma = 1). Re-tune on your machine with tune.py and pass the result with --set.
"""
import copy

_BASE = dict(eta=1e-3, N=3, n_s="auto", n_c=5, r=0.05, tau=100, xi=0.5, gamma0=1.0,
             max_iters=20000, log_grad_error=True)


def _make(H=50, seeds=10, eps=0.5, tau=100, adj_horizon=20, adj_restarts=None, ell=1,
          adapt_adj_horizon=100,
          eta_single=1e-3, eta_multi=1e-3, eta_sub=1e-2,
          eta_lora=1e-2, eta_finetune=1e-3, eta_scratch=1e-3, n_down=3,
          ns_lora=1, ns_finetune=1, ns_scratch=1, r_finetune=None, r_lora=None, nc_adapt=20,
          tune_etas=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2), tune_ns=("auto",), **base):
    b = copy.deepcopy(_BASE)
    b.update(tau=tau, **base)
    return dict(
        H=H, seeds=seeds, n_test=200, eps=eps,
        adj_restarts=adj_restarts or ell + 2, adj_horizon=adj_horizon, autodiff="numpy",
        base=b,
        single=dict(eta=eta_single),
        multi=dict(eta=eta_multi),
        subspace=dict(eta=eta_sub),
        adapt=dict(n_down=n_down, adj_horizon=adapt_adj_horizon,
                   lora=dict(eta=eta_lora, n_s=ns_lora, r=r_lora or b["r"], n_c=nc_adapt),
                   finetune=dict(eta=eta_finetune, n_s=ns_finetune, r=r_finetune or b["r"], n_c=nc_adapt),
                   scratch=dict(eta=eta_scratch, n_s=ns_scratch, n_c=nc_adapt)),
        tune=dict(etas=list(tune_etas), ns=list(tune_ns)),
    )


CONFIGS = {
    "pendulum": _make(ell=1, tau=100, eps=0.5,
                      eta_single=1e-2, eta_multi=1e-2, eta_sub=1e-2,
                      eta_lora=3e-3, eta_finetune=1e-4, eta_scratch=1e-3, r_finetune=0.01,
                      tune_etas=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1)),
    "cartpole": _make(ell=3, tau=100, eps=0.5,
                      eta_single=5e-6, eta_multi=5e-6, eta_sub=5e-5,
                      eta_lora=5e-6, eta_finetune=3e-7, eta_scratch=5e-6,
                      ns_scratch=3, r_finetune=0.01,
                      tune_etas=(1e-7, 1e-6, 1e-5, 1e-4, 3e-4)),
    "launch": _make(ell=2, tau=100, eps=0.5,
                    eta_single=3e-3, eta_multi=3e-3, eta_sub=3e-3,
                    eta_lora=1e-4, eta_finetune=3e-5, eta_scratch=1e-3, r_finetune=0.01, r_lora=0.01,
                    tune_etas=(1e-3, 3e-3, 1e-2, 3e-2)),
    "flexarm": _make(ell=1, tau=150, eps=0.5, adj_horizon=30, adapt_adj_horizon=300, nc_adapt=50,
                     eta_single=1e-4, eta_multi=1e-4, eta_sub=1e-2,
                     eta_lora=1e-3, eta_finetune=1e-5, ns_finetune=10, r_finetune=0.01,
                     eta_scratch=3e-5, ns_scratch="auto",
                     tune_etas=(3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)),
}


def quick(cfg):
    """Small smoke-test version of a config (few systems, seeds and iterations)."""
    c = copy.deepcopy(cfg)
    c.update(H=min(c["H"], 10), seeds=2, n_test=50)
    c["base"]["max_iters"] = min(c["base"]["max_iters"], 3000)
    if c.get("adapt"):
        c["adapt"]["n_down"] = 1
    return c
