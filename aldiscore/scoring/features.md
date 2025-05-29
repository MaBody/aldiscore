# Dataset Features
= Features describing the unaligned sequences.
These features can be used to predict the MSA difficulty as they don't require the computation of a multiple sequence alignment.

Given a set of unaligned features (i.e. a `Dataset`), you can generate the set of features as pandas DataFrame like this:

```python
import pathlib

from features.dataset import Dataset
from features.features import get_dataset_features

sequence_file = pathlib.Path("path/to/sequences.fasta")

dataset = Dataset(sequence_file)
features = get_dataset_features(dataset)
```


The resulting Pandas DataFrame has a single row only as it summarizes all features across all sequences of the dataset. 
It contains the following features:
- `avg_perc_hash_hamming_{16/256}bit`: Average perceptual hash hamming distance between all pairs of sequences in the
        dataset using a hash size of 16/256.
- `{min/max/mean/median/std}_seq_length_taxa_ratio`: The ratio of the shortest/longest/average/median sequence to the number of sequences (`{min/max/mean/median}_seq_len / #sequences`).
- `seq_length_ratio`: The ratio of the smallest to the longest sequence (`min_seq_len / max_seq_len`).
- `std_seq_length_taxa_ratio`: The ratio of the standard deviation of sequence lengths to the number of sequences (`std_seq_len / #sequences`).
- `lower_bound_gap_percentage`: Minimum expected percentage of gaps based on the lengths of the unaligned sequences. This lower bound is reached if the longest sequence remains without gaps after the MSA computation. (`1 - mean_seq_len / max_seq_len`).
- `{min/max/mean/std}_entropy`: Minimum/maximum/average/standard deviation Shannon entropy of all sequences.
- `basic_{min/max/mean/std}_score_ratio`: Minimum/maximum/average/standard deviation of pairwise alignment score ratios (see below) using the basic pairwise alignment penalties (`(1, 0, 0, 0)`).
- `advanced_{min/max/mean/std}_score_ratio`: Minimum/maximum/average/standard deviation of pairwise alignment score ratios (see below) using the advanced pairwise alignment penalties (`(2, -0.5, 0, -3)`, according to Chowdhury and Garai (2017)).
- `basic_{min/max/mean/std}_gap_ratio`: Minimum/maximum/average/standard deviation of pairwise alignment gap ratios (see below) using the basic pairwise alignment penalties (`(1, 0, 0, 0)`).
- `advanced_{min/max/mean/std}_gap_ratio`: Minimum/maximum/average/standard deviation of pairwise alignment gap ratios (see below) using the advanced pairwise alignment penalties (`(2, -0.5, 0, -3)`, according to Chowdhury and Garai (2017)).
- `basic_{min/max/mean/std}_inflation_ratio`: Minimum/maximum/average/standard deviation of pairwise alignment inflation ratios (see below) using the basic pairwise alignment penalties (`(1, 0, 0, 0)`).
- `advanced_{min/max/mean/std}_inflation_ratio`: Minimum/maximum/average/standard deviation of pairwise alignment inflation ratios (see below) using the advanced pairwise alignment penalties (`(2, -0.5, 0, -3)`, according to Chowdhury and Garai (2017)).
- `basic_{min/max/mean/std/cv}_avg_gap_length`: Minimum/maximum/average/standard deviation/coefficient of variation of pairwise alignment average gap lengths (see below) using the basic pairwise alignment penalties (`(1, 0, 0, 0)`).
- `advanced_{min/max/mean/std/cv}_avg_gap_length`: Minimum/maximum/average/standard deviation/coefficient of variation of pairwise alignment average gap lengths (see below) using the advanced pairwise alignment penalties (`(2, -0.5, 0, -3)`, according to Chowdhury and Garai (2017)).
- `kmer_similarity_15`: K-mer similarity score (see below) for `k=15`.
- `kmer_similarity_25`: K-mer similarity score (see below) for `k=25`.

### Score ratio
Proportion of the pairwise alignment score explained by the length of the smaller sequence. Computed as $\frac{score(PA(s_1, s_2))}{min(len(s_1), len(s_2))}$.

### Gap ratio
Proportion of gaps in the pairwise alignment, i.e. the number of gaps divided by the total number of characters.

### Inflation ratio
Increase in length of the longest unaligned sequence compared to the pairwise alignment. Computed as $\frac{max(len(s_1), len(s_2))}{len(PA(s_1, s_2))}$

### Average gap length
Given an aligned sequence (i.e. a sequence potentially containing gaps), computes the average length of a sequence of consecutive gaps in the sequence. Gap sequences at the beginning/end of a sequence are not considered.
Example:
1. Aligned sequence: `--AG---CT-A-G--`
2. Trim gaps at the beginning/ end: `AG---CT-A-G`
3. Count the length for each consecutive sequence of gaps: (3, 1, 1)
4. Compute the average: `(3 + 1 + 1) / 3 = 1.667`

### K-Mer similarity score
Estimates the similarity of the unaligned sequences using the k-mer similarity. This score is computed as follows:
1. Choose a random set of 1000 k-mers from the set of all possible k-mers of all unaligned sequences.
2. For each of these k-mers: compute the fraction of other sequences containing this k-mer.
3. Average the resulting fractions.


# Ensemble Features
= Features describing a set of replicate MSAs (= an `Ensemble`) computed for the same set of unaligned features (= `Dataset`).


The resulting Pandas DataFrame has a single row only as it summarizes all features across all MSAs of the ensemble. 
It contains the following features:
- `avg_perc_hash_hamming_16bit`: Average perceptual hash hamming distance between all pairs of MSAs in the ensemble using a hash size of 16.
- `avg_perc_hash_hamming_256bit`: Average perceptual hash hamming distance between all pairs of MSAs in the ensemble using a hash size of 256.
- `lc`: Letter confidence (Muscle5 EFA statistic)
- `cc`: Column confidence (Muscle5 EFA statistic)
- `dispersion`: Ensemble dispersion (Muscle5 EFA statistic)
- `avg_min_inflation`: Average minimum inflation of sequence length.
- `avg_max_inflation`: Average maximum inflation of sequence length.
- `{min/max/mean/std}_normalized_sp_score`: Minimum/maximum/mean/standard deviation of the normalized sum-of-pairs score over all MSAs in the ensemble (see below).
- **`pythia difficulty` features: All features based on the Pythia difficulty prediction.  See `Ensemble.pythia_features` for the list of features.  Note that each feature in the pythia difficulty features is summarized using three statistics: mean, standard deviation, and median. Depending on the set used version of Pythia, the set of features may differ. For Pythia > v1.1.0, the set of features comprises:
  - `num_taxa`: Number of sequences in the MSA.
  - `{min/max/mean/std}_num_sites`: Minimum/maximum/average/standard deviation of the number of sites of all MSAs.
  - `{min/max/mean/std}_num_patterns`: Minimum/maximum/average/standard deviation of the number of unique sites of all MSAs.
  - `{min/max/mean/std}_num_patterns/num_taxa`: Minimum/maximum/average/standard deviation of the patterns-over-taxa ratios.
  - `{min/max/mean/std}_num_sites/num_taxa`: Minimum/maximum/average/standard deviation of the sites-over-taxa ratios.
  - `{min/max/mean/std}_proportion_gaps`: Minimum/maximum/average/standard deviation of the proportion of gaps in all MSAs.
  - `{min/max/mean/std}_proportion_invariant`: Minimum/maximum/average/standard deviation of the proportion of invariant sites in all MSAs.
  - `{min/max/mean/std}_entropy`: Minimum/maximum/average/standard deviation of Shannon Entropy of all MSAs. The Entropy for one MSA is computed as average column-wise Shannon Entropy.
  - `{min/max/mean/std}_bollback`: Minimum/maximum/average/standard deviation of the Bollback Multinomial test statistic of all MSAs.
  - `{min/max/mean/std}_avg_rfdist_parsimony`: Minimum/maximum/average/standard deviation of the average Robinson-Foulds distance between 100 parsimony trees inferred for each of the MSAs.
  - `{min/max/mean/std}_proportion_unique_topos_parsimony`: Minimum/maximum/average/standard deviation of the fraction of unique tree topologies among 100 parsimony trees inferred for each of the MSAs.
  - `{min/max/mean/std}_pythia_difficulty`: Minimum/maximum/average/standard deviation of the predicted difficulties of all MSAs.

### Normalized sum-of-pairs score
To compute the sum-of-pairs score for an MSA, we first count the number of matching characters between all unique pairs of sequences (note that two gaps do not count as a match).
We then compute the average sum-of-pairs score of all $\frac{N(N-1)}{2}$ pairwise scores.
Finally, we normalize using the number of sites of the MSA as this is the maximum possible score.
