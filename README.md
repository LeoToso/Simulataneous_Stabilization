# Learning to Simultaneously Stabilize can be Sample-Efficient — experiments

Python code for the numerical experiments of the paper. It replaces the original notebooks
(kept in the repository for reference), with two changes:

* **two-point** zeroth-order gradient estimation (Algorithm 1) instead of the one-point estimator;
* **no process noise**: rollouts are `x_{t+1} = (A + B K) x_t` from random resets.

Each system has one command that runs every experiment and writes all figures.

```bash
pip install -r requirements.txt

python run.py --system pendulum  --out results/pendulum
python run.py --system cartpole  --out results/cartpole
python run.py --system launch    --out results/launch
python run.py --system flexarm   --out results/flexarm

bash run_all.sh --seeds 10 --workers 16                      # all of them
```

Options: `--seeds` (default 10), `--workers` (parallel seeds, default all cores), `--H`
(number of training systems, default 50), `--no-adapt`, `--quick` (smoke test),
`--set key=value` (override any config entry, e.g. `--set single.eta=1e-3 --set base.tau=200`),
`--replot` (redraw figures from `results/<system>/results.pkl`), `--verbose`.

## Systems (`simstab/systems.py`)

| name | model | d_X | d_U | unstable modes ℓ |
|---|---|---|---|---|
| `pendulum` | inverted pendulum, Euler dt = 0.25, random mass/length (as in the notebook) | 2 | 1 | 1 |
| `cartpole` | cart-pole, Euler dt = 0.25, random masses/length, Q = 100 I, R = 0.1 I (as in the notebook) | 4 | 1 | 3 (see below) |
| `launch` | launch-vehicle pitch plane (Wie, Du, Whorton 2008), ZOH dt = 0.25 (as in the notebook) | 3 | 1 | 2 |
| `flexarm` | flexible arm, 20-section Hencky bar chain (Lutkus, Lindemann, Tu, arXiv:2610.40931, Sec. 6.1), ZOH dt = 0.05 | 40 | 1 | 1 |

* Cart-pole: the Euler double integrator of the cart gives a double eigenvalue at exactly 1.
  As in the notebook, the "unstable" subspace counts |λ| ≥ 1, so ℓ = 3 (pendulum mode + cart).
* Flexible arm: the arm paper does not print A and B, so they are rebuilt from its model:
  uniform bars (m̄ = 0.04 kg, ℓ = 0.05 m), K0 stiffnesses spaced in [1, 1.5], κ midway
  between the two largest generalized eigenvalues of (SᵀdiagwS, K0), Rayleigh damping
  (c0, c1) = (0.4, 0.02), ZOH at 0.05 s, B normalized to ‖B‖ = 1. The nominal arm reproduces
  the reported spectrum exactly (unstable 1.129; slowest stable 0.874 and 0.581 ± 0.633j;
  ‖B‖ = 2.14 before normalization). Tasks perturb section length (±3%), mass (±10%), damping
  (±10%) and the position of κ in its interval (40–60%); the unstable eigenvalue ranges over
  1.115–1.145 and every task keeps 39 stable modes (slowest about 0.88).
* A three-unstable-mode arm (the arm paper's Sec. 6.4) can be built with
  `flexible_arm(n_unstable=3, unstable_targets=(1.15, 1.12, 1.09))`, which reproduces that
  spectrum exactly. It is not included as an experiment: with one input (or 3–4 joint torques)
  the three buckling modes are nearly jointly uncontrollable, the optimal reduced LQR cost is
  10⁶–10⁸, and the discount schedule (α ≈ q / 2J per iteration) would need about 10⁶ iterations.
* `downstream` regime (adaptation experiment): a shifted parameter range per system
  (shorter/heavier pendulum, heavier pole, max-q-like launch conditions, heavier and less
  damped arm). Only downstream systems that the common controller K_c does **not** stabilize
  are used.

## What is run (`simstab/experiments.py`)

For each seed, H training systems are drawn and:

1. **All modes, single task** — Algorithm 2 on system 1 alone.
2. **All modes, multitask** — Algorithm 2 on the H systems (shared averaged gradient).
3. **Unstable modes, multitask** — Algorithm 3 (adjoint sampling, m = ℓ + 2 restarts,
   horizon T) per system, projector averaging, then Algorithm 4 with K = θ Φ̂ᵀ.
4. **Adaptation** — K_c from (3). For downstream systems not stabilized by K_c:
   * **low-rank adaptation** (Algorithm 5): residual representation Ψ̂ of A + B K_c by adjoint
     sampling with the rank ℓ̄_h supplied, then the discount method on θ_h ∈ R^{d_U × ℓ̄_h},
     K_h = K_c + θ_h Ψ̂ᵀ, cost (96);
   * **full-rank fine-tuning**: the same discount method from K_c on all d_U d_X entries;
   * **from scratch**: Algorithm 2 on the downstream system with K_0 = 0.

   Both methods that start from K_c use γ_{h,0} = 0.9 / ρ(A + B K_c)², and from scratch uses
   0.9 / ρ(A)².

Side experiments: the gradient-estimation error versus H for the full and the subspace
parametrizations (Lemma .9), and the subspace distance d(Φ̂, Φ) versus the adjoint horizon T
(Theorem .3).

Exact model quantities (Lyapunov solutions, eigenvalues) are used **only** for logging:
spectral radii, true gradients, the ground-truth subspace, and the n_s rule below.

### Sample counting and protocol (`simstab/configs.py`)

Samples are state transitions collected from **each** system, as in Corollaries .1/.4:
every discount iteration costs `(2 N n_s + n_c) τ` per system. Adjoint (VJP) evaluations of
Algorithm 3 (m T per system) are reported separately, as in eq. (91).

All settings share N = 3, n_c = 5, r = 0.05, ξ = 0.5, τ and γ_0 = 0.9 / max_h ρ(A_h)².

* **Gradient batch n_s ("auto")** — matched to a common gradient accuracy ε (relative RMS error
  at initialization): n_s = ⌈(e₁/ε)²⌉, where e₁ is the error with one pair per system. This is
  the batch rule of Lemma .9 (H n_s ≳ d/ε²). It is how collaboration (H) and the shared subspace
  (d = d_U ℓ instead of d_U d_X) enter the sample complexity. The chosen n_s is stored per run.
* **Step size η** — per setting, since the admissible step depends on the parametrization's
  Lipschitz constant. The values in `configs.py` are the most sample-efficient ones found by
  `tune.py` (each tuning seed had to stabilize). Re-tune on your machine with:

```bash
python tune.py --system flexarm --seeds 3 --workers 16                          # main settings
python tune.py --system flexarm --settings lora,finetune,scratch --ns 1,3,10 --rs 0.05,0.01
```

  then pass the printed `--set ...` options to `run.py` or edit `configs.py`.
* **Adaptation**: fixed n_s per method (1 by default) with tuned η; full-rank fine-tuning also
  needs a smaller smoothing radius (`r = 0.01`). With r = 0.05 its two-point perturbations of
  K_c leave the damped-stable neighbourhood (the condition r√d < ν_K fails), and the estimate is
  badly biased. Low-rank adaptation does not have this problem, since it only perturbs along the
  residual unstable directions.

## Outputs (`results/<system>/`)

| file | content |
|---|---|
| `fig_discount_vs_samples.pdf` / `_vs_iterations` | γ_j for the three settings (median, 10–90% band over seeds) |
| `fig_spectral_radius_vs_samples.pdf` / `_vs_iterations` | ρ(A⁽¹⁾ + B⁽¹⁾ K_j) |
| `fig_spectral_radius_max_vs_*.pdf` | max_h ρ(A⁽ʰ⁾ + B⁽ʰ⁾ K_j) over the H training systems |
| `fig_grad_error_vs_iterations.pdf` | ‖∇̂J − ∇J‖_F along the run (relative version also written) |
| `fig_grad_error_vs_H.pdf` | relative gradient error vs H, full vs subspace parametrization |
| `fig_subspace_vs_T.pdf` | d(Φ̂, Φ) vs adjoint horizon |
| `fig_sample_complexity.pdf` | samples per system to reach γ = 1, per setting and seed |
| `fig_adaptation.pdf` | γ and ρ(A + B K_{h,j}) of the downstream systems: low-rank vs full-rank vs scratch |
| `summary.md`, `summary.json` | success rates, samples, iterations, held-out stabilization, n_s, adaptation |
| `results.pkl`, `config.json` | raw logs and the exact configuration |

## Code layout

```
run.py                  one command per system
tune.py                 step-size / batch grid search
run_all.sh              every system
simstab/systems.py      system families
simstab/algorithms.py   Algorithms 1-5 (model-free)
simstab/lqr.py          exact quantities for evaluation only
simstab/experiments.py  experiment definitions
simstab/plotting.py     figures and summaries
simstab/configs.py      hyperparameters
```
