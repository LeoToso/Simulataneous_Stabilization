"""Figures and summary tables from the per-seed results."""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "single":   dict(color="purple",     label="All modes, single task"),
    "multi":    dict(color="orange",     label="All modes, multitask $H={H}$"),
    "subspace": dict(color="tab:blue",   label="Unstable modes, multitask $H={H}$"),
    "lora":     dict(color="tab:green",  label=r"Low-rank adaptation of $K_c$ (rank $\bar\ell_h$)"),
    "finetune": dict(color="tab:red",    label=r"Full-rank fine-tuning of $K_c$"),
    "scratch":  dict(color="tab:gray",   label="From scratch (single task)"),
}
MAIN = ("single", "multi", "subspace")
ADAPT = ("lora", "finetune", "scratch")

plt.rcParams.update({"font.size": 14, "axes.grid": True, "grid.alpha": 0.4,
                     "legend.fontsize": 12, "lines.linewidth": 2.5, "savefig.bbox": "tight"})


def _label(key, H):
    return STYLE[key]["label"].format(H=H)


def _step_interp(x_run, y_run, grid):
    """Value of a logged curve at each grid point (last logged value at or before it; the
    final value is held after termination, the initial value before the first update)."""
    idx = np.maximum(np.searchsorted(x_run, grid, side="right") - 1, 0)
    y = np.where(idx >= 0, y_run[np.clip(idx, 0, None)], np.nan)
    return y


def _band(runs, field, grid, xfield="samples"):
    Y = np.array([_step_interp(np.asarray(r[xfield]), np.asarray(r[field]), grid) for r in runs])
    return np.nanmedian(Y, 0), np.nanpercentile(Y, 10, 0), np.nanpercentile(Y, 90, 0)


def _iter_band(runs, field, n, hold=True):
    Y = []
    for r in runs:
        y = np.asarray(r[field], float)
        pad = y[-1] if hold and not field.startswith("grad") else np.nan
        Y.append(np.concatenate([y, np.full(max(0, n - len(y)), pad)])[:n])
    Y = np.array(Y)
    return np.nanmedian(Y, 0), np.nanpercentile(Y, 10, 0), np.nanpercentile(Y, 90, 0)


def _curve(ax, runs, key, field, H, xaxis, xgrid=None, n_iter=None):
    if not runs:
        return
    st = STYLE[key]
    if xaxis == "samples":
        med, lo, hi = _band(runs, field, xgrid)
        x = xgrid
    else:
        med, lo, hi = _iter_band(runs, field, n_iter)
        x = np.arange(n_iter)
    ax.plot(x, med, color=st["color"], label=_label(key, H))
    if len(runs) > 1:
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.2, lw=0)


def _sample_grid(groups):
    """Log-spaced grid; the initial controller (0 samples) is drawn at the left edge."""
    xs = [np.asarray(r["samples"]) for runs in groups for r in runs]
    lo = min(x[x > 0].min() for x in xs if np.any(x > 0))
    hi = max(x.max() for x in xs)
    return np.geomspace(lo / 3, hi * 1.05, 800)


def _finish(ax, field, xaxis, logx=True):
    if field == "gamma":
        ax.axhline(1, ls="--", color="k", lw=2.5)
        ax.set_ylabel(r"Discount factor $\gamma_j$")
        ax.set_ylim(0, 1.04)
    elif field == "rho1":
        ax.axhline(1, ls="--", color="k", lw=2.5)
        ax.set_ylabel(r"$\rho(A^{(1)}+B^{(1)}K_j)$")
    elif field == "rho_max":
        ax.axhline(1, ls="--", color="k", lw=2.5)
        ax.set_ylabel(r"$\max_h\,\rho(A^{(h)}+B^{(h)}K_j)$")
    elif field == "grad_err":
        ax.set_ylabel("Relative gradient error")
        ax.set_yscale("log")
    elif field == "grad_err_abs":
        ax.set_ylabel(r"$\|\widehat\nabla J^{\gamma_j} - \nabla J^{\gamma_j}\|_F$")
        ax.set_yscale("log")
    if xaxis == "samples":
        ax.set_xlabel("Samples per system (state transitions)")
        if logx:
            ax.set_xscale("log")
    else:
        ax.set_xlabel("Discount iterations $j$")


def plot_main(results, outdir, H):
    groups = {k: [r[k] for r in results if k in r] for k in MAIN}
    grid = _sample_grid(groups.values())
    n_iter = max(len(r["gamma"]) for runs in groups.values() for r in runs)
    figs = {}
    for field, name in (("gamma", "discount"), ("rho1", "spectral_radius"),
                        ("rho_max", "spectral_radius_max")):
        for xaxis in ("samples", "iterations"):
            fig, ax = plt.subplots(figsize=(10, 6))
            for k in MAIN:
                _curve(ax, groups[k], k, field, H, xaxis, grid, n_iter)
            _finish(ax, field, xaxis)
            ax.legend()
            fn = f"fig_{name}_vs_{xaxis}.pdf"
            fig.savefig(os.path.join(outdir, fn))
            fig.savefig(os.path.join(outdir, fn.replace(".pdf", ".png")), dpi=150)
            plt.close(fig)
            figs[(field, xaxis)] = fn
    # gradient estimation error along the iterations (absolute and relative)
    for field, fn in (("grad_err_abs", "fig_grad_error_vs_iterations"),
                      ("grad_err", "fig_grad_rel_error_vs_iterations")):
        if not all(field in r for runs in groups.values() for r in runs):
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for k in MAIN:
            _curve(ax, groups[k], k, field, H, "iterations", n_iter=n_iter)
        _finish(ax, field, "iterations")
        ax.legend()
        fig.savefig(os.path.join(outdir, fn + ".pdf"))
        fig.savefig(os.path.join(outdir, fn + ".png"), dpi=150)
        plt.close(fig)
    # samples to stabilize
    fig, ax = plt.subplots(figsize=(8, 5))
    tot = [[r["total_samples"] for r in groups[k] if r["stabilized"]] for k in MAIN]
    pos = np.arange(len(MAIN))
    for i, k in enumerate(MAIN):
        if tot[i]:
            ax.bar(i, np.median(tot[i]), color=STYLE[k]["color"], alpha=0.85)
            ax.scatter(np.full(len(tot[i]), i), tot[i], color="k", s=12, zorder=3)
    ax.set_xticks(pos, [_label(k, H).replace(", ", ",\n") for k in MAIN], fontsize=11)
    ax.set_yscale("log")
    vals = [v for t in tot for v in t]
    if vals:
        ax.set_ylim(min(vals) / 10, max(vals) * 2)
    ax.set_ylabel("Samples per system to reach $\\gamma=1$\n(bar: median, dots: seeds)")
    fig.savefig(os.path.join(outdir, "fig_sample_complexity.pdf"))
    fig.savefig(os.path.join(outdir, "fig_sample_complexity.png"), dpi=150)
    plt.close(fig)


def plot_adaptation(results, outdir):
    groups = {k: [run for r in results if "adapt" in r for run in r["adapt"][k] if run] for k in ADAPT}
    if not any(groups.values()):
        return
    grid = _sample_grid([g for g in groups.values() if g])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for field, ax in zip(("gamma", "rho1"), axes):
        for k in ADAPT:
            _curve(ax, groups[k], k, field, 1, "samples", grid)
        _finish(ax, field, "samples")
        if field == "rho1":
            ax.set_ylabel(r"$\rho(A^{(h)}+B^{(h)}K_{h,j})$, downstream system")
    axes[0].legend(loc="lower right")
    fig.savefig(os.path.join(outdir, "fig_adaptation.pdf"))
    fig.savefig(os.path.join(outdir, "fig_adaptation.png"), dpi=150)
    plt.close(fig)


def plot_grad_vs_H(data, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))
    H = np.asarray(data["H"])
    for key, st in (("full", STYLE["multi"]), ("subspace", STYLE["subspace"])):
        y = np.asarray(data[key])
        lbl = "All modes ($d_U d_X$ parameters)" if key == "full" else r"Unstable subspace ($d_U\ell$ parameters)"
        ax.plot(H, y, marker="o", color=st["color"], label=lbl)
        if key + "_q25" in data:
            ax.fill_between(H, data[key + "_q25"], data[key + "_q75"], color=st["color"], alpha=0.2, lw=0)
        ax.plot(H, y[0] / np.sqrt(H / H[0]), ls=":", color=st["color"], lw=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of systems $H$")
    ax.set_ylabel("Relative gradient error ($n_s=1$ per system)\nmean, 25-75% band")
    ax.legend(title=r"dotted: $\propto 1/\sqrt{H}$")
    fig.savefig(os.path.join(outdir, "fig_grad_error_vs_H.pdf"))
    fig.savefig(os.path.join(outdir, "fig_grad_error_vs_H.png"), dpi=150)
    plt.close(fig)


def plot_subspace_vs_T(data, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))
    T, D = np.asarray(data["T"]), np.maximum(np.asarray(data["dist"]), 1e-16)
    ax.semilogy(T, np.median(D, 0), color="tab:blue")
    ax.fill_between(T, D.min(0), D.max(0), color="tab:blue", alpha=0.2, lw=0)
    ax.set_xlabel("Adjoint horizon $T$")
    ax.set_ylabel(r"$d(\widehat\Phi,\Phi)$  (median, min-max over seeds)")
    fig.savefig(os.path.join(outdir, "fig_subspace_vs_T.pdf"))
    fig.savefig(os.path.join(outdir, "fig_subspace_vs_T.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _stats(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return dict(median=float(np.median(vals)), mean=float(np.mean(vals)), min=float(np.min(vals)),
                max=float(np.max(vals)))


def summarize(results, cfg, outdir, extra=None):
    S = dict(system=results[0]["system"], H=cfg["H"], seeds=[r["seed"] for r in results],
             config=cfg, settings={})
    for k in MAIN:
        runs = [r[k] for r in results]
        S["settings"][k] = dict(
            success_rate=float(np.mean([r["stabilized"] for r in runs])),
            samples_per_system=_stats([r["total_samples"] for r in runs if r["stabilized"]]),
            discount_iterations=_stats([r["iterations"] for r in runs if r["stabilized"]]),
            final_train_rho_max=_stats([r["train_rho_max"] for r in runs]),
            heldout_stabilized_fraction=_stats([r["heldout_frac"] for r in runs]),
            subspace_err=_stats([r.get("subspace_err") for r in runs]) if k == "subspace" else None,
        )
    if any("adapt" in r for r in results):
        A = dict()
        for k in ADAPT:
            runs = [run for r in results if "adapt" in r for run in r["adapt"][k] if run]
            A[k] = dict(n=len(runs),
                        success_rate=float(np.mean([x["stabilized"] for x in runs])) if runs else None,
                        samples=_stats([x["total_samples"] for x in runs if x["stabilized"]]),
                        discount_iterations=_stats([x["iterations"] for x in runs if x["stabilized"]]))
        A["lora_subspace_err"] = _stats([x.get("subspace_err") for r in results if "adapt" in r
                                         for x in r["adapt"]["lora"] if x])
        A["ell_bar"] = [int(v) for r in results if "adapt" in r for v in r["adapt"]["ell_bar"]]
        A["rho_c"] = [float(v) for r in results if "adapt" in r for v in r["adapt"]["rho_c"]]
        A["n_downstream_found"] = [int(r["adapt"]["n_found"]) for r in results if "adapt" in r]
        S["adaptation"] = A
    if extra:
        S.update(extra)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(S, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    lines = [f"# {S['system']}  (H = {cfg['H']}, {len(results)} seeds)", "",
             "| setting | success | samples/system (median) | discount iters (median) | "
             "max_h rho final (median) | held-out stabilized (median) |",
             "|---|---|---|---|---|---|"]
    for k in MAIN:
        s = S["settings"][k]
        f = lambda d, fmt: (fmt % d["median"]) if d else "-"
        lines.append(f"| {_label(k, cfg['H'])} | {s['success_rate']:.2f} | "
                     f"{f(s['samples_per_system'], '%.3g')} | {f(s['discount_iterations'], '%.0f')} | "
                     f"{f(s['final_train_rho_max'], '%.3f')} | {f(s['heldout_stabilized_fraction'], '%.2f')} |")
    if "adaptation" in S:
        A = S["adaptation"]
        lines += ["", "## Adaptation to downstream systems not stabilized by K_c", "",
                  f"rank ell_bar per system: {A['ell_bar']}", "",
                  "| method | runs | success | samples (median) | discount iters (median) |",
                  "|---|---|---|---|---|"]
        for k in ADAPT:
            a = A[k]
            if a["n"]:
                lines.append(f"| {_label(k, 1)} | {a['n']} | {a['success_rate']:.2f} | "
                             f"{(a['samples'] or {}).get('median', float('nan')):.3g} | "
                             f"{(a['discount_iterations'] or {}).get('median', float('nan')):.0f} |")
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return S
