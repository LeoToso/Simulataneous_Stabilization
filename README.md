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

