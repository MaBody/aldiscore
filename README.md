# AlDiScore - Alignment Difficulty Score

AlDiScore provides two approaches for quantifying multiple sequence alignment (MSA) difficulty:


1. **Predictive Scoring**: Predict alignment difficulty from unaligned sequences using ML. Supports both nucleotides and amino-acids sequences. 
2. **Heuristic Scoring**: Compute dispersion within an ensemble of alternative alignments. Requires the pre-computed MSA ensemble. 

## Setup

Clone this repository and navigate to the top folder:
```shell
git clone git@github.com:MaBody/aldiscore.git
cd aldiscore
```

Then pick one of the two supported setups.

### a) uv (PyPI)

```shell
uv sync
uv run aldiscore -h
```

`uv sync` creates `.venv` with the locked dependencies from `uv.lock` and installs `aldiscore` in editable mode. Optional extras: `uv sync --extra pythia`, `--extra train`, `--extra demo`.

> **macOS:** `parasail` has no arm64 wheel on PyPI, so `uv sync` compiles it from source, and the `lightgbm` wheel needs OpenMP at runtime. Either install these first (`brew install autoconf automake libtool libomp`) or use the pixi setup below, which ships prebuilt binaries.

### b) pixi (conda-forge / bioconda)

```shell
pixi install
pixi run aldiscore -h
```

Compiled dependencies (`parasail`, `lightgbm`) come prebuilt from conda. Additional environments:

```shell
pixi install -e pythia   # adds pythiaphylopredictor + raxml-ng
pixi install -e demo     # adds ipykernel for the demo notebook
```

### Tests

```shell
uv run pytest    # or: pixi run test
```

## Command Line Interface

The `aldiscore` command line tool supports both heuristic scoring and prediction:


### Prediction
```shell
# Predict difficulty for unaligned sequences
aldiscore predict path/to/sequences.fasta --in-type AA
aldiscore predict path/to/alignment.fasta # aligned input works too, gaps are removed
aldiscore predict path/to/sequences.phy --in-format=phylip

```
Note that it detects by default the datatype (AA or DNA) but you can also specify it manually.

### Heuristic Scoring

Note that our Prediction models are trained with d_pos pairwise score. 

```shell
# Compute pairwise scores (d_ssp, d_seq, d_pos, d_phash)
aldiscore heuristic path/to/ensemble/ --method d_pos
# Compute a pairwise distance matrix between the sequences
aldiscore heuristic path/to/ensemble/ --method d_pos --out-type matrix

# Compute set-based scores (conf_set, conf_entropy, conf_displace)
aldiscore heuristic path/to/ensemble/ --method conf_entropy --out-type scalar
```

For detailed help:
```shell
aldiscore -h
aldiscore predict -h
aldiscore heuristic -h
```

## Python Library
### Prediction
```python
from aldiscore.prediction.predictor import DifficultyPredictor

# Initialize predictor with a pre-trained model ("aa" or "dna")
predictor = DifficultyPredictor(model="aa")

# Predict difficulty for sequences
score = predictor.predict("path/to/sequences.fasta")
```

### Heuristic Scoring
```python
from aldiscore.datastructures import Ensemble
from aldiscore.scoring import pairwise, set_based

# Load ensemble of alternative alignments
ensemble = Ensemble.load("path/to/ensemble/")

# Compute pairwise score (d_pos) --> Default
d_pos = pairwise.DPosDistance().compute(ensemble)

# Compute confusion score (conf_entropy)
conf_ent = set_based.ConfusionEntropy().compute(ensemble)
```

We recommend checking out [demo.ipynb](demo/demo.ipynb) for a quick and intuitive overview of the library. The demo notebook requires ipykernel in the environment:

```shell
uv sync --extra demo    # or: pixi install -e demo
```

## Input Data

We build our implementation on top of BioPython data classes (Seq, SeqRecord, MultipleSeqAlignment) to support different file types. 

For the heuristics, we use our own wrapper classes Alignment, Dataset, and Ensemble. We need these wrappers to implement sorting and caching strategies.

- `Alignment` contains `Bio.Align.MultipleSeqAlignment`
- `Dataset` cotains `list[Bio.SeqRecord.SeqRecord]`
- `Ensemble` contains `list[Alignment]` and `Dataset`

## Background

### Heuristic Methods

We provide implementations for seven scoring methods that compute the dispersion within an ensemble of alignments.
These scores quantify the uncertainty in the alignment process by analyzing variability between alternative alignments.

| Pairwise                          | Description                                                |
| --------------------------------- | ---------------------------------------------------------- |
| $\text{d}_{\text{SSP}}$ [[1]](#1) | Homology set metric. Ignores gaps.                         |
| $\text{d}_{\text{seq}}$ [[1]](#1) | Homology set metric. Identical coding of gaps in sequence. |
| $\text{d}_{\text{pos}}$ [[1]](#1) | Homology set metric. Identical coding of consecutive gaps. |

| Set-based                | Description                                               |
| ------------------------ | --------------------------------------------------------- |
| $\text{Conf}_{Set}$      | Number of unique entries per replication set.             |
| $\text{Conf}_{Entropy}$  | Shannon entropy per replication set.                      |
| $\text{Conf}_{Displace}$ | Binned standard deviation of indices per replication set. |

Our preferred uncertainty quantification method is the pairwise $\text{d}_{\text{pos}}$ score.

### Prediction Model

The prediction functionality allows estimating alignment difficulty directly from unaligned sequences, without the need to compute alternative alignments. This is achieved through a machine learning model that was trained on over 9,000 MSA datasets of DNA (5800) and AA (3851) sequences sets. For the labels, we used the d_pos metric on a diverse ensemble of 48 alignments. Regarding model performance, we report an R^2=0.885 in amino-acid sequences and R^2=0.836 in nucleotide sequences.

Key features:
- Fast prediction without alignment computation
- Support for both DNA and protein sequences

The prediction pipeline consists of two main components:
1. Feature extraction (sequence properties, k-mer statistics, etc.)
2. Model inference using pre-trained LightGBM models



## References

<a id="1">[1]</a>
Blackburne, B. P., & Whelan, S. (2012).
Measuring the distance between multiple sequence alignments.
Bioinformatics, 28(4), 495-502.
