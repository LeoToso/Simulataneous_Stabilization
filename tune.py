#!/usr/bin/env python3
"""Grid search of (eta, n_s) for each setting of one system.

For every setting (single / multi / subspace, and the adaptation methods lora / finetune /
scratch, which first compute K_c with the subspace method on each tuning seed), each
(eta, n_s) pair is run on several seeds. The selected pair is the one with the fewest median
samples per system among the pairs that stabilized on EVERY seed. All other hyperparameters are
shared, so each method is compared at its own best step size and batch size.

    python tune.py --system flexarm --seeds 3 --workers 12
    python tune.py --system flexarm --settings single,multi --etas 1e-4,3e-4,1e-3 --ns auto
    python tune.py --system pendulum --settings lora,finetune,scratch --ns 1,2,5,10

Writes results/<system>/tuning.json and prints the selected values in the --set format of run.py.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from simstab import lqr  # noqa: E402
from simstab.algorithms import Problem, discount_method, shared_representation, adapt_low_rank  # noqa: E402
from simstab.configs import CONFIGS  # noqa: E402
from simstab.experiments import _cfg, auto_gamma0  # noqa: E402
from simstab.systems import get_family  # noqa: E402


ADAPT = ("lora", "finetune", "scratch")


def common_controller(system, cfg, seed, n_down=2):
    """K_c from the shared-subspace method (config hyperparameters) and downstream systems
    that K_c does not stabilize, for tuning the adaptation methods."""
    fam = get_family(system)
    rng = np.random.default_rng(seed)
    A, B, Q, R = fam.sample(rng, cfg["H"], "train")
    g0 = auto_gamma0(max(lqr.spectral_radius(a) for a in A), cfg["base"].get("gamma0", 1.0))
    Phi, _ = shared_representation(A, fam.ell, cfg["adj_restarts"], cfg["adj_horizon"], rng)
    c = _cfg(cfg["base"], cfg["subspace"], gamma0=g0, eps=cfg["eps"], log_grad_error=False)
    Kc = discount_method(Problem(A, B, Q, R, Phi=Phi, project_cost=True), c, rng)["K"]
    downs = []
    while len(downs) < n_down:
        Ad, Bd, Qd, Rd = fam.sample(rng, 1, "downstream")
        if lqr.spectral_radius(Ad[0] + Bd[0] @ Kc) >= 1:
            downs.append((Ad[0], Bd[0], Qd[0], Rd[0]))
    return Kc, downs


def adapt_run(system, cfg, setting, eta, n_s, r, seed, Kc, downs):
    fam = get_family(system)
    rng = np.random.default_rng(seed + 1)
    res = []
    for A, B, Q, R in downs:
        Ac = A + B @ Kc
        if setting == "scratch":
            c = _cfg(cfg["base"], {k: v for k, v in cfg["adapt"][setting].items() if k == "n_c"},
                     gamma0=auto_gamma0(lqr.spectral_radius(A), 1.0), eta=eta,
                     n_s=n_s, r=r, eps=cfg["eps"], log_grad_error=False)
            o = discount_method(Problem(A[None], B[None], Q[None], R[None]), c, rng)
        else:
            c = _cfg(cfg["base"], {k: v for k, v in cfg["adapt"][setting].items() if k == "n_c"},
                     gamma0=auto_gamma0(lqr.spectral_radius(Ac), 1.0), eta=eta,
                     n_s=n_s, r=r, eps=cfg["eps"], log_grad_error=False)
            if setting == "lora":
                ell_bar = lqr.num_unstable(Ac)
                T = cfg["adapt"].get("adj_horizon", cfg["adj_horizon"])
                o = adapt_low_rank(A, B, Q, R, Kc, ell_bar, c, max(cfg["adj_restarts"], ell_bar), T, rng)
            else:
                o = discount_method(Problem(A[None], B[None], Q[None], R[None], Kbase=Kc[None]), c, rng)
        res.append(o)
    return dict(stabilized=all(o["stabilized"] for o in res),
                samples=float(np.median([o["total_samples"] for o in res])),
                iters=[o["iterations"] for o in res], status=[o["status"] for o in res],
                n_s=[o["n_s"] for o in res])


def one_run(system, cfg, setting, eta, n_s, r, seed, Kc=None, downs=None):
    if setting in ADAPT:
        return adapt_run(system, cfg, setting, eta, n_s, r, seed, Kc, downs)
    fam = get_family(system)
    rng = np.random.default_rng(seed)
    A, B, Q, R = fam.sample(rng, cfg["H"], "train")
    g0 = auto_gamma0(max(lqr.spectral_radius(a) for a in A), cfg["base"].get("gamma0", 1.0))
    c = _cfg(cfg["base"], cfg.get(setting), gamma0=g0, eta=eta, n_s=n_s, r=r, eps=cfg["eps"],
             log_grad_error=False)
    if setting == "single":
        p = Problem(A[:1], B[:1], Q[:1], R[:1])
    elif setting == "multi":
        p = Problem(A, B, Q, R)
    else:
        Phi, _ = shared_representation(A, fam.ell, cfg["adj_restarts"], cfg["adj_horizon"], rng)
        p = Problem(A, B, Q, R, Phi=Phi, project_cost=True)
    o = discount_method(p, c, rng)
    return dict(stabilized=o["stabilized"], samples=o["total_samples"], iters=o["iterations"],
                status=o["status"], n_s=o["n_s"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--settings", default="single,multi,subspace")
    ap.add_argument("--etas", default=None, help="comma separated (default: config tune grid)")
    ap.add_argument("--ns", default=None, help="comma separated (default: config tune grid)")
    ap.add_argument("--rs", default=None, help="smoothing radii, comma separated (default: config r)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--first-seed", type=int, default=100, help="tuning seeds differ from run.py seeds")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = copy.deepcopy(CONFIGS[args.system])
    grid = cfg.get("tune", {})
    etas = [float(x) for x in args.etas.split(",")] if args.etas else grid["etas"]
    nss = [x if x == "auto" else int(x) for x in args.ns.split(",")] if args.ns else grid["ns"]
    rs = [float(x) for x in args.rs.split(",")] if args.rs else [cfg["base"]["r"]]
    settings = args.settings.split(",")
    seeds = range(args.first_seed, args.first_seed + args.seeds)

    jobs = list(itertools.product(settings, etas, nss, rs, seeds))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        ctrl = {}
        if any(s in ADAPT for s in settings):
            print("computing the common controllers K_c for the adaptation settings ...", flush=True)
            futs = {sd: ex.submit(common_controller, args.system, cfg, sd) for sd in seeds}
            ctrl = {sd: f.result() for sd, f in futs.items()}
        futs = [ex.submit(one_run, args.system, cfg, s, e, n, r, sd,
                          *(ctrl[sd] if s in ADAPT else (None, None))) for s, e, n, r, sd in jobs]
        res = [f.result() for f in futs]
    table = {}
    for (s, e, n, r, sd), o in zip(jobs, res):
        table.setdefault((s, e, n, r), []).append(o)

    out = dict(system=args.system, H=cfg["H"], seeds=list(seeds), grid={}, selected={})
    for s in settings:
        print(f"\n== {s}")
        best = None
        for e, n, rr in itertools.product(etas, nss, rs):
            runs = table[(s, e, n, rr)]
            ok = all(o["stabilized"] for o in runs)
            med = float(np.median([o["samples"] for o in runs]))
            print(f"  eta={e:<8.1e} n_s={str(n):<4s} r={rr:<6g} ({[o['n_s'] for o in runs]}) ok={str(ok):5s} "
                  f"median samples={med:10.3g} iters={[o['iters'] for o in runs]} "
                  f"status={[o['status'] for o in runs]}")
            out["grid"][f"{s}|{e}|{n}|{rr}"] = runs
            if ok and (best is None or med < best[3]):
                best = (e, n, rr, med)
        if best:
            out["selected"][s] = dict(eta=best[0], n_s=best[1], r=best[2], median_samples=best[3])
    print("\nselected:")
    for s, v in out["selected"].items():
        pre = f"adapt.{s}" if s in ADAPT else s
        print(f"  --set {pre}.eta={v['eta']} --set {pre}.n_s={v['n_s']} --set {pre}.r={v['r']}"
              f"   (median {v['median_samples']:.3g} samples)")
    path = args.out or os.path.join("results", args.system, "tuning.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
