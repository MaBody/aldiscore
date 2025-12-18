# AlDiScore - Alignment Difficulty Score

AlDiScore provides two approaches for quantifying multiple sequence alignment (MSA) difficulty:

1. **Heuristic Scoring**: Compute dispersion within an ensemble of alternative alignments
2. **Predictive Scoring**: Predict alignment difficulty from unaligned sequences using ML

## Features

- Command-line interface for heuristics and prediction
- Multiple scoring methods for ensemble analysis
- Pre-trained models for difficulty prediction
- Supports DNA and amino acid sequences

## Setup 
1. Clone this repository and navigate to the top folder.
```shell
git clone git@github.com:MaBody/aldiscore.git
```


### a) Library + CLI 

2. build with python+setuptools and install distribution:
   ```shell
   python -m build
   pip install dist/aldiscore-<version>.whl
   ```
### b) Library on Conda

2. Use the `environment.yml` file to set up a conda environment:
   ```shell
   conda env create -f environment.yml
   conda activate aldiscore
   pip install rich_argparse
   pip install .
   ```

## Command Line Interface

The `aldiscore` command line tool supports both heuristic scoring and prediction:


### Prediction
```shell
# Predict difficulty for unaligned sequences
aldiscore predict path/to/sequences.fasta
aldiscore predict path/to/alignment.fasta --drop-gaps
aldiscore predict path/to/sequences.phy --in-format=phylip

```

### Heuristic Scoring
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

# Initialize predictor with pre-trained model
predictor = DifficultyPredictor()

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

We recommend checking out [demo.ipynb](demo/demo.ipynb) for a quick and intuitive overview of the library. The demo notebook requires ipykernel to be installed in the environment:

```shell
conda install ipykernel
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

The prediction functionality allows estimating alignment difficulty directly from unaligned sequences, without the need to compute alternative alignments. This is achieved through a machine learning model that was trained on over 11,000 MSA datasets of DNA and AA sequences. For the labels, we used the d_pos metric on a diverse ensemble of 48 alignments. Regarding model performance, we report an RMSE of 0.04.

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
