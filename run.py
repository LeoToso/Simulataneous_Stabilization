#!/usr/bin/env python3
"""Run all experiments for one system and write the figures.

    python run.py --system pendulum   --out results/pendulum
    python run.py --system cartpole   --out results/cartpole
    python run.py --system launch     --out results/launch
    python run.py --system flexarm    --out results/flexarm

Options: --seeds 10 --workers 10 --H 50 --quick (small smoke test) --verbose
         --set single.eta=1e-3 --set base.tau=200 ... (override any config entry)
         --replot (redraw figures from results.pkl)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

from simstab import plotting  # noqa: E402
from simstab.configs import CONFIGS, quick  # noqa: E402
from simstab.experiments import run_seed, grad_error_vs_H, subspace_vs_T  # noqa: E402


def apply_set(cfg, items):
    for item in items or []:
        key, val = item.split("=", 1)
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (default from config)")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--H", type=int, default=None, help="number of training systems")
    ap.add_argument("--quick", action="store_true", help="small smoke test")
    ap.add_argument("--no-adapt", action="store_true", help="skip the adaptation experiment")
    ap.add_argument("--set", action="append", help="override, e.g. single.eta=1e-3")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = copy.deepcopy(CONFIGS[args.system])
    if args.quick:
        cfg = quick(cfg)
    if args.H:
        cfg["H"] = args.H
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.no_adapt:
        cfg["adapt"] = None
    cfg = apply_set(cfg, args.set)
    out = args.out or os.path.join("results", args.system + ("_quick" if args.quick else ""))
    os.makedirs(out, exist_ok=True)
    pkl = os.path.join(out, "results.pkl")

    if args.replot:
        with open(pkl, "rb") as f:
            saved = pickle.load(f)
        results, extra, cfg = saved["results"], saved["extra"], saved["config"]
    else:
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        seeds = list(range(args.first_seed, args.first_seed + cfg["seeds"]))
        t0 = time.time()
        results = []
        print(f"[{args.system}] H={cfg['H']}  seeds={seeds}  workers={args.workers}  -> {out}", flush=True)
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(run_seed, args.system, cfg, s, args.verbose): s for s in seeds}
            f_gh = ex.submit(grad_error_vs_H, args.system, cfg, 0)
            f_sT = ex.submit(subspace_vs_T, args.system, cfg, range(5))
            for fu in as_completed(futs):
                r = fu.result()
                results.append(r)
                msg = "  ".join(f"{k}: {'ok' if r[k]['stabilized'] else 'FAIL'} "
                                f"{r[k]['total_samples']:.3g} samples" for k in plotting.MAIN)
                print(f"  seed {r['seed']} done in {r['time']:.0f}s | {msg}", flush=True)
            extra = dict(grad_error_vs_H=f_gh.result(), subspace_vs_T=f_sT.result())
        results.sort(key=lambda r: r["seed"])
        with open(pkl, "wb") as f:
            pickle.dump(dict(results=results, extra=extra, config=cfg), f)
        print(f"[{args.system}] all runs finished in {time.time() - t0:.0f}s", flush=True)

    plotting.plot_main(results, out, cfg["H"])
    plotting.plot_adaptation(results, out)
    plotting.plot_grad_vs_H(extra["grad_error_vs_H"], out)
    plotting.plot_subspace_vs_T(extra["subspace_vs_T"], out)
    plotting.summarize(results, cfg, out, dict(grad_error_vs_H=extra["grad_error_vs_H"]))
    with open(os.path.join(out, "summary.md")) as f:
        print(f.read())
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
