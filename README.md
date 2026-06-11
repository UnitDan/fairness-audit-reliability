# Certifiable Auditing

This repository contains the method and experiments for certifiable auditing of fairness assessments.  The code computes robustness radii for group-level fairness conclusions, trains the Adult/MLP and MovieLens-1M/LightGCN models used in the experiments, and reproduces the synthetic and empirical r-epsilon studies.

## Repository Layout

- `opt_for_r.py`: core robustness-radius optimizer.
- `pipeline_adult_mlp.py`: trains MLP checkpoints on the Adult dataset.
- `pipeline_ml1m_lightgcn.py`: trains LightGCN checkpoints on MovieLens-1M.
- `exp_analytic/`: synthetic analytic comparison experiments.
- `exp_misjudge/`: r-epsilon frontier experiments for synthetic, Adult, and MovieLens-1M settings.
- `exp_phiandr/`: analysis for the relation between baseline fairness gaps and robust radii.
- `models_mlp/`: small Adult/MLP checkpoints used by the empirical experiments.
- `scripts/download_data.py`: helper for downloading third-party datasets into the expected local layout.
- `DATA.md`: dataset sources, citations, and redistribution notes.

Generated pickle files, large LightGCN checkpoints, raw third-party datasets, notebooks, and local logs are intentionally not tracked in the open-source version.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the CPU or CUDA build of PyTorch that matches your machine if the default wheel is not appropriate.

## Data

The repository does not redistribute Adult or MovieLens-1M raw data.  Download them with:

```bash
python scripts/download_data.py --adult --ml1m
```

This creates the expected `adult/` and `ml-1m/` directories.  See `DATA.md` for citations and license/terms notes.

## Reproducing Experiments

Run the synthetic analytic comparison:

```bash
cd exp_analytic
python main.py --n_m_configs 10 --n_p_configs 10 --n_eps_configs 10 --k 1000 --transform normalized --random_seed 42
```

Train the Adult MLP checkpoints:

```bash
python pipeline_adult_mlp.py
```

Train the MovieLens-1M LightGCN checkpoints:

```bash
python pipeline_ml1m_lightgcn.py
```

Run r-epsilon experiments:

```bash
cd exp_misjudge
python main.py --dataset synthetic --n_samples 1000 --eps_points 51 --verbose
python main.py --dataset adult --n_samples 1000 --model_ratio 0.5 --eps_points 51 --verbose
python main.py --dataset ml1m --n_samples 1000 --model_ratio 0.5 --eps_points 51 --verbose
```

For the full sweep used in the paper-style experiments:

```bash
cd exp_analytic && bash run.sh
cd ../exp_misjudge && bash run.sh
```

The full sweeps can take a long time and will write ignored result caches under `exp_analytic/k=*/` and `exp_misjudge/results/`.

## Notes for Public Release

- Keep raw datasets and large generated results out of Git history.  If exact cached artifacts are needed, attach them as a GitHub Release asset or archive them separately.
- The snapshot branch `codex/open-source-materials-snapshot` preserves the full local experiment state before this cleanup.
- Add the final paper citation and project license before making the repository public.
