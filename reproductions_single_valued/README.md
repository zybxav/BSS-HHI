# Reproduction package

For the single-valued setting, we made our best effort to reproduce the related
algorithms based on publicly available code and the original papers. For
methods with open-source implementations, we made only the necessary
adaptations while preserving their core mechanisms. For methods without
publicly available implementations, we reproduced their core functionality
according to the descriptions in the original papers.

## Included methods

End-to-end HHI:

- `treehist-paper-topk`: paper-based TreeHist with a finite-sample top-k
  candidate cap.
- `bitstogram-core`: Bitstogram discovery and independent verification.
- `hadaheavy-core`: HadaHeavy hierarchical search using HadaOracle.
- `privateexpander-core`: both PrivateExpanderSketch size regimes with bounded
  finite-domain list recovery.

Frequency oracles:

- `olh`, `oue`, and `sue`.
- `cms` and `hcms`: Apple CMS/HCMS adapted from pure-LDP's public hash family
  and reconstruction.
- `hadaoracle-core`.
- `pgr-python-port`: the MSS authors' Python PGR port.
- `mss`: the authors' MSS implementation with aggregate sufficient statistics.
- `pan-ocms-independent`.

## Third-party code

Apple CMS and HCMS are adapted from
[pure-LDP](https://github.com/Samuel-Maddock/pure-LDP) (commit
`d4b2827b49127119b710271e48ac59710872b531`).

MSS and the Python implementation of PGR are adapted from the authors'
[private-frequency-oracle-rns](https://github.com/hharcolezi/private-frequency-oracle-rns)
repository (commit `824a3f72e360bba45bfb59047f5dab9fd7b2d3cc`). The Python
PGR implementation is itself based on the official
[PGR C++ implementation](https://github.com/minilek/private_frequency_oracles)
(commit `f043e0c3ba180e05ce0883e93e0863c699599e17`).

The vendored MSS subset retains its upstream MIT license, and the PGR-derived
files retain their BSD 3-Clause notices. The pure-LDP adaptation is attributed
here under pure-LDP's MIT license. All other reproduced methods in this package
are independent implementations based on their respective papers.

Primary references: [TreeHist and Bitstogram](https://jmlr.csail.mit.edu/papers/volume21/18-786/18-786.pdf),
[Apple CMS/HCMS](https://machinelearning.apple.com/research/learning-with-privacy-at-scale),
[PrivateExpanderSketch](https://arxiv.org/abs/1711.04740),
[HadaHeavy/HadaOracle](https://proceedings.mlr.press/v151/wu22e.html),
[PGR](https://proceedings.mlr.press/v162/feldman22a.html),
[MSS](https://github.com/hharcolezi/private-frequency-oracle-rns), and
[Pan-OCMS](https://arxiv.org/abs/2603.11523).

## Environment

```powershell
conda env create -f environment.yml
conda activate ldp-reproductions
```

## Prepare data

The source format is one user per line, with positive 1-based item IDs
separated by spaces or commas. Each non-empty user contributes one uniformly
sampled distinct item. All methods use the same converted file and truth.

```powershell
python prepare_data.py --input kosarak=D:\data\kosarak.txt --output-dir prepared_data
```

Repeat `--input NAME=PATH` to prepare several datasets in one command. The
default conversion seed is 20260724; the default 100,000-user subsampling seed
is 20260725.

## Run experiments

```powershell
python run_experiments.py --data-dir prepared_data --output-dir run_output
```

The default grid uses epsilon values 1, 1.5, 2, 2.5, 3, 3.5, and 4, with five
mechanism seeds. A focused run can select a family and methods:

```powershell
python run_experiments.py --data-dir prepared_data --families HHI `
  --hhi-methods treehist-paper-topk bitstogram-core --epsilons 2 --seeds 101
```

Input files remain 1-based. The mechanism boundary converts IDs to 0-based,
and reported HHI IDs are mapped back to 1-based. The runner writes only records
that it actually executes.
