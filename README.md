# BSS-HHI

This repository is the implementation of the paper: **Mitigating Unreliable Ranking for Heavy Hitter Identification in Large Domains under Local Differential Privacy**.

## File structure

```text
BSS-HHI/
├── bsshhi/                 BSS-HHI mechanisms and modules.
│   ├── algorithm.py        Binary candidate reconstruction and low-rank estimation.
│   ├── protocol.py         The end-to-end BSS-HHI protocol.
│   ├── svim.py             SVIM frequent-item identification.
│   └── case_studies.py     Relation-mining and federated-HHI implementations.
├── experiments/
│   ├── paper.py            The runner for the paper experiments E1–E18.
│   ├── configs/            The configurations for E1–E18.
│   └── settings.json       The selected parameters for each dataset.
├── baselines/
│   ├── main/               Baselines used in the main experiments.
│   ├── case_studies/       The LDP-RM relation-mining baseline.
│   ├── hhi/                HHI baselines discussed in the appendix.
│   ├── fo/                 Frequency oracles discussed in the appendix.
│   ├── common/             Shared baseline utilities.
│   ├── vendor/             Third-party baseline dependencies and licenses.
│   └── README.md           Baseline methods, sources, and environment.
├── data/                   Datasets and experiment inputs.
├── run_paper.py             Command-line entry for E1–E18.
├── run.py                   Command-line entry for custom BSS-HHI runs.
├── run_baseline.py          Command-line entry for the comparison methods.
├── requirements.txt         Python dependencies for BSS-HHI.
└── DATA.md                  Dataset files and input formats.
```

See [baselines/README.md](baselines/README.md) for the comparison methods, their locations, and commands.

## Environment

Use **Python 3.8** and install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Datasets

The datasets are provided in `data/` as gzip-compressed text files, which the Python runners read directly.

| Dataset | Location |
|---|---|
| Kosarak | `data/main/kosarak.txt.gz` |
| MovieLens | `data/main/movielens.txt.gz` |
| UserBehavior | `data/main/UserBehavior1300.txt.gz` |
| Zipf-65K | `data/main/syn65k.txt.gz` |
| Zipf-131K | `data/main/syn130k.txt.gz` |
| IFTTT | `data/case_studies/ifttt_100k.txt.gz` |
| FedUB | `data/case_studies/ub1_party1.txt.gz`, `ub1_party2.txt.gz`, `ub1_party3.txt.gz`, and `ub_p123.txt.gz` |

The single-valued inputs are in `data/single/`, and the user-scale and Zipf-exponent inputs are in `data/supplement/`; the complete file list is in [DATA.md](DATA.md).

A new dataset is a UTF-8 text file with one user per line and positive integer item IDs separated by spaces:

```text
1 7 12
2 7
3 9 12 15
```

Save it as `data/main/mydata.txt` and select it with `--dataset mydata`; the item domain is inferred from its IDs.

## Run the paper experiments

From the repository root, run:

```bash
python run_paper.py E1
```

Replace `E1` with the experiment ID below; each configuration runs 10 times using the parameters in `experiments/configs/`.

| ID | Experiment |
|---|---|
| E1 | Utility under different privacy budgets |
| E2 | Utility under different top-k values |
| E3 | User-group allocation |
| E4 | Conditional-expansion sequences |
| E5 | Relation mining on IFTTT |
| E6 | Federated heavy-hitter identification on FedUB |
| E7 | Candidate recall across phases |
| E8 | Single-valued heavy-hitter identification |
| E9 | SVD ablation |
| E10 | Candidate capacities |
| E11 | Phase 3 retention ratio |
| E12 | Low-rank threshold |
| E13 | Number of SVD iterations |
| E14 | Number of participating users |
| E15 | Zipf exponent |
| E16 | Original and fully random ID encodings |
| E17 | Gradual ID permutation |
| E18 | Low-rank structure under score ordering |

## Run BSS-HHI with custom parameters

Select a dataset, privacy budget, top-k, and number of users:

```bash
python run.py --dataset kosarak --epsilon 2 --top-k 32 --users 100000
```

Table 1 parameters can be supplied as follows:

```bash
python run.py --dataset kosarak --epsilon 2 --top-k 32 --users 100000 --c 2,1,1 --x1-ratio 0.05 --c0 50 --rho2 0.3 --theta 0.9 --iterations 5
```

| Argument | Description |
|---|---|
| `--dataset` | Dataset name: `kosarak`, `movielens`, `syn65k`, `syn130k`, `userbehavior1300`, or a custom name. |
| `--epsilon` | Privacy budget ε. |
| `--top-k` | Number of heavy hitters, default 32. |
| `--users` | Number of participating users, default 100000. |
| `--c` | Conditional-expansion sequence, such as `2,1,1`, or `none` for Phase 1 candidate reconstruction. |
| `--x1-ratio` | Phase 1 candidate-domain ratio, where `0.05` means 5%. |
| `--c0` | Initial per-partition candidate capacity C0. |
| `--rho2` | Phase 3 retention ratio ρ2. |
| `--theta` | Low-rank threshold θ. |
| `--iterations` | Number of SVD iterations T. |

The provided datasets use their selected settings by default; for a custom dataset, supply `--c`, `--x1-ratio`, `--c0`, and `--rho2`:

```bash
python run.py --dataset mydata --epsilon 2 --top-k 32 --users 100000 --c 2,1,1 --x1-ratio 0.05 --c0 50 --rho2 0.3
```

## Run the comparison methods

Install their dependencies:

```bash
python -m pip install -r baselines/requirements.txt
```

Select a method with the same dataset, privacy-budget, top-k, and user-count arguments:

```bash
python run_baseline.py --method svim --dataset kosarak --epsilon 2 --top-k 32 --users 100000
python run_baseline.py --method svdfo --dataset kosarak --epsilon 2 --top-k 32 --users 100000
python run_baseline.py --method pemset --dataset kosarak --epsilon 2 --top-k 32 --users 100000
```

Several methods can share the same sampled input in one invocation:

```bash
python run_baseline.py --method svim svdfo pemset --dataset kosarak --epsilon 2 --top-k 32 --users 100000
```

The appendix HHI and FO methods use the same command:

```bash
python run_baseline.py --method treehist --dataset syn65k --epsilon 2 --top-k 32 --users 100000
python run_baseline.py --method olh --dataset syn65k --epsilon 2 --top-k 32 --users 100000
```

SVIM, SVD-FO, and PEMSET use set-valued inputs. The appendix HHI and FO methods use the provided single-valued inputs in `data/single/`; new datasets are converted to one item per user. The method IDs and source locations are listed in [baselines/README.md](baselines/README.md).

ARBSF uses MATLAB; its entry point is `baselines/main/ARBSF/work8_long_s/run_example.m`, with data preparation and configuration described in [its README](baselines/main/ARBSF/README.md).

## Results

Paper experiment results are saved to `results/E1.csv` through `results/E18.csv`, with the corresponding metrics' means and standard deviations; E15 also writes `results/E15_profile.csv`.

Custom runs print the dataset, privacy budget, top-k, user count, and F1/NCR/MCR to the terminal.

The comparison runner prints each method's parameters, F1/NCR/MCR, and running time to the terminal.
