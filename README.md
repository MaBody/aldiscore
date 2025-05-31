# AlDiScore - Alignment Dispersion Score

## Setup
1. Use the `environment.yml` file to set up the conda environment:
    ```shell
    conda env create -f environment.yml
    ```
2. Activate the new conda environment:
    ```shell
    conda activate aldiscore
    ```
   
## Input Data
This library builds on BioPython data classes (Seq, SeqRecord, MultipleSeqAlignment). We implement our own wrapper classes Alignment, Dataset, and Ensemble:
- Alignment: ``BioPython.Align.MultipleSeqAlignment`` + some more program logic 
- Dataset: ``list[SeqRecord]`` (unaligned sequences) + some more program logic
- Ensemble: ``list[Alignment]`` and the respective Dataset

## Scoring
We provide implementations for six scoring methods that compute the dispersion within an ensemble of alignments. 
The scores can be used as a proxy for the uncertainty/difficulty of the alignment step.


|Pairwise     | Description                                               |
|--------------|-----------------------------------------------------------|
|d_SSP  | Homology set metric. Ignores gaps.|
|d_seq  | Homology set metric. Identical coding of gaps in sequence.|
|d_pos  | Homology set metric. Identical coding of consecutive gaps.|
|pHash  | Perceptual hashing of alignment matrix.|


|Set-based     | Description                                               |
|--------------|-----------------------------------------------------------|
|Column Confidence  | Fraction of reproduced columns across ensemble.           |
|ConfSet            | Number of unique entries per replication set.             |
|ConfEntropy        | Shannon entropy per replication set.                      |
|ConfDisplace       | Binned standard deviation of indices per replication set. |

Our preferred method is the pairwise d_pos score.

## Demo 
We recommend checking out [demo.ipynb](notebooks/demo.ipynb) for a quick and intuitive overview of the functionalities.