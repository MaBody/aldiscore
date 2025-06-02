# AlDiScore - Alignment Dispersion Score

This package implements several measures of dispersion within an ensemble of multiple sequence alignments of the same sequencing dataset.

## Setup

1. Clone this repository.
2. Use the `environment.yml` file to set up the conda environment:
   ```shell
   conda env create -f environment.yml
   ```
3. Activate the new conda environment:
   ```shell
   conda activate aldiscore
   ```
4. Install aldiscore locally:
   ```shell
   pip install .
   ```

## Input Data

We build our implementation on top of BioPython data classes (Seq, SeqRecord, MultipleSeqAlignment). For convenience, we implement our own wrapper classes Alignment, Dataset, and Ensemble. We need these wrappers to implement sorting and caching strategies.

- `Alignment` contains `Bio.Align.MultipleSeqAlignment`
- `Dataset` cotains `list[Bio.SeqRecord.SeqRecord]`
- `Ensemble` contains `list[Alignment]` and `Dataset`

## Scoring

We provide implementations for six scoring methods that compute the dispersion within an ensemble of alignments.
The scores can be used as a proxy for the uncertainty/difficulty of the alignment step.

| Pairwise                          | Description                                                |
| --------------------------------- | ---------------------------------------------------------- |
| $\text{d}_{\text{SSP}}$ [[1]](#1) | Homology set metric. Ignores gaps.                         |
| $\text{d}_{\text{seq}}$ [[1]](#1) | Homology set metric. Identical coding of gaps in sequence. |
| $\text{d}_{\text{pos}}$ [[1]](#1) | Homology set metric. Identical coding of consecutive gaps. |
| pHash [[2]](#2)                   | Perceptual hashing of alignment matrix.                    |

| Set-based                | Description                                               |
| ------------------------ | --------------------------------------------------------- |
| $\text{Conf}_{Set}$      | Number of unique entries per replication set.             |
| $\text{Conf}_{Entropy}$  | Shannon entropy per replication set.                      |
| $\text{Conf}_{Displace}$ | Binned standard deviation of indices per replication set. |

Our preferred uncertainty quantification method is the pairwise $\text{d}_{\text{pos}}$ score.

## Demo

We recommend checking out [demo.ipynb](demo/demo.ipynb) for a quick and intuitive overview of the provided functionality. The demo notebook requires ipykernel to be installed in the environment:

```shell
conda install ipykernel
```

## References

<a id="1">[1]</a>
Blackburne, B. P., & Whelan, S. (2012).
Measuring the distance between multiple sequence alignments.
Bioinformatics, 28(4), 495-502.

<a id="2">[2]</a>
Zauner, C. (2010).
Implementation and benchmarking of perceptual image hash functions.
PhD Thesis
