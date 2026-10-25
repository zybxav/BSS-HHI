# Baselines

This directory contains the baselines used in the main experiments and case studies, along with the HHI and frequency-oracle methods discussed in the appendix.

## Main experiments: `main/`

| Method | Runner ID | Location | Description |
|---|---|---|---|
| SVIM | `svim` | `../bsshhi/svim.py` | Set-valued frequent-item identification under local differential privacy. |
| SVD-FO | `svdfo` | `main/SVD_FO/` | Two-dimensional candidate reconstruction and SVD, called through the ordered-coordinate adapter in `adapter.py`. |
| PEMSET | `pemset` | `main/PEMSET/` | Prefix-based set-valued heavy-hitter identification with MWheel. |
| ARBSF | MATLAB | `main/ARBSF/work8_long_s/` | The MATLAB implementation with its `run_example.m` entry point. |

## Case studies

| Method | Location | Description |
|---|---|---|
| LDP-RM | `case_studies/LDP_RM/` | The relation mining method LDP-RM. |
| TAPS | — | Federated heavy-hitter analytics, with the implementation obtained by contacting the authors. |

## Appendix HHI methods: `hhi/`

| Method | Runner ID | Location | Description |
|---|---|---|---|
| TreeHist | `treehist` | `hhi/treehist_paper/` | Paper-based TreeHist with a finite-sample top-k candidate cap. |
| Bitstogram | `bitstogram` | `hhi/bitstogram_core/` | Bitstogram candidate discovery, decoding, and independent verification. |
| HadaHeavy | `hadaheavy` | `hhi/hadaheavy_core/` | Hierarchical heavy-hitter identification using HadaOracle. |
| PrivateExpanderSketch | `privateexpander` | `hhi/private_expander_sketch_core/` | The Hashtogram full-domain branch and a finite Reed–Solomon list-recovery adaptation. |

## Appendix frequency oracles: `fo/`

| Method | Runner ID | Location | Description |
|---|---|---|---|
| OLH, OUE, SUE | `olh`, `oue`, `sue` | `fo/standard/` | Standard local hashing and unary-encoding frequency oracles. |
| Apple CMS/HCMS | `cms`, `hcms` | `fo/apple_cms/` | Count-Mean and Hadamard Count-Mean sketches adapted from pure-LDP. |
| HadaOracle | `hadaoracle` | `fo/hadaoracle/` | A paper-based Hadamard frequency oracle with median aggregation. |
| PGR | `pgr` | `fo/pgr/` | The MSS authors' Python port of Projective Geometry Response. |
| MSS | `mss` | `fo/mss/` | Modular Subset Selection adapted from the authors' implementation. |
| Pan-OCMS | `pan-ocms` | `fo/pan_ocms/` | A paper-based implementation of the OCMS mechanism. |
| Hashtogram | `hashtogram` | `fo/hashtogram/` | The frequency-estimation component used by the HHI reproductions. |

`common/` contains shared implementation utilities, and `vendor/` contains third-party dependencies and their licenses.

## Environment

Install the baseline dependencies in the Python 3.8 environment from the repository root:

```bash
python -m pip install -r baselines/requirements.txt
```

## Run a method

From the repository root, select a method ID from the tables above:

```bash
python run_baseline.py --method svim --dataset kosarak --epsilon 2 --top-k 32 --users 100000
python run_baseline.py --method treehist --dataset syn65k --epsilon 2 --top-k 32 --users 100000
python run_baseline.py --method olh --dataset syn65k --epsilon 2 --top-k 32 --users 100000
```

Several methods can share the same sampled input in one invocation:

```bash
python run_baseline.py --method svim svdfo pemset --dataset kosarak --epsilon 2 --top-k 32 --users 100000
```

SVIM, SVD-FO, and PEMSET use the set-valued datasets in `data/main/`; the appendix HHI and FO methods use `data/single/`, with new datasets converted to one item per user. Dataset names and custom input formats are described in the [repository README](../README.md).

The runner prints JSON containing the method, parameters, F1/NCR/MCR, and running time; redirect it to a file to save the results:

```bash
python run_baseline.py --method svim --dataset kosarak --epsilon 2 --top-k 32 --users 100000 > svim_kosarak.json
```

## Run ARBSF

Set the dataset paths and parameters at the top of `main/ARBSF/work8_long_s/run_example.m`, then run `run_example` in MATLAB from that directory; results are saved to `experiment_runner/results/`.

## References

| Method | Original paper |
|---|---|
| SVIM | Wang, Li, and Jha. [Locally Differentially Private Frequent Itemset Mining](https://ieeexplore.ieee.org/document/8418600). IEEE S&P, 2018. |
| SVD-FO / LDP-RM | Dong et al. [Relation Mining Under Local Differential Privacy](https://www.usenix.org/conference/usenixsecurity24/presentation/dong-kai). USENIX Security, 2024. |
| PEMSET | Zhu et al. [Heavy Hitter Identification over Large-Domain Set-Valued Data with Local Differential Privacy](https://doi.org/10.1109/TIFS.2023.3324726). IEEE TIFS, 19:414–426, 2024. |
| ARBSF | Du et al. [Top-k Discovery Under Local Differential Privacy: An Adaptive Sampling Approach](https://doi.org/10.1109/TDSC.2024.3471923). IEEE TDSC, 22(2):1763–1780, 2025. |
| TreeHist, Bitstogram, Hashtogram | Bassily et al. [Practical Locally Private Heavy Hitters](https://jmlr.org/papers/v21/18-786.html). JMLR, 21(16):1–42, 2020. |
| PrivateExpanderSketch | Bun, Nelson, and Stemmer. [Heavy Hitters and the Structure of Local Privacy](https://arxiv.org/abs/1711.04740). ACM Transactions on Algorithms, 2019. |
| HadaHeavy, HadaOracle | Wu and Wirth. [Asymptotically Optimal Locally Private Heavy Hitters via Parameterized Sketches](https://proceedings.mlr.press/v151/wu22e.html). AISTATS, 2022. |
| OLH, OUE, SUE | Wang et al. [Locally Differentially Private Protocols for Frequency Estimation](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/wang-tianhao). USENIX Security, 2017. |
| Apple CMS/HCMS | Apple Differential Privacy Team. [Learning with Privacy at Scale](https://machinelearning.apple.com/research/learning-with-privacy-at-scale). Apple Machine Learning Research, 2017. |
| PGR | Feldman et al. [Private Frequency Estimation via Projective Geometry](https://proceedings.mlr.press/v162/feldman22a.html). ICML, 2022. |
| MSS | Arcolezi. [Private Frequency Estimation via Residue Number Systems](https://ojs.aaai.org/index.php/AAAI/article/view/40862). AAAI, 2026. |
| Pan-OCMS | Pan. [Strict Optimality of Frequency and Distribution Estimation Under Local Differential Privacy](https://arxiv.org/abs/2603.11523). arXiv:2603.11523, 2026. |
| TAPS | Zhang, Ye, and Hu. Federated Heavy Hitter Analytics with Local Differential Privacy. Proceedings of the ACM on Management of Data, 3(1):1–27, 2025. |

## Code sources

For the single-valued methods, public implementations are adapted where available, and paper-based implementations reproduce the corresponding core mechanisms.

| Component | Source |
|---|---|
| LDP-RM | [Original implementation](https://github.com/zybxav/BSS-HHI/blob/fa698de6e12931e2e5ee6135885e10e734da6a8c/BSS/ldp_rm.py), with its required modules packaged in `case_studies/LDP_RM/`. |
| Apple CMS/HCMS | [pure-LDP](https://github.com/Samuel-Maddock/pure-LDP), commit `d4b2827b49127119b710271e48ac59710872b531`, MIT license. |
| MSS and Python PGR | [private-frequency-oracle-rns](https://github.com/hharcolezi/private-frequency-oracle-rns), commit `824a3f72e360bba45bfb59047f5dab9fd7b2d3cc`, with the upstream MIT and BSD notices retained. |
| Original PGR | [PGR C++ implementation](https://github.com/minilek/private_frequency_oracles), commit `f043e0c3ba180e05ce0883e93e0863c699599e17`. |

