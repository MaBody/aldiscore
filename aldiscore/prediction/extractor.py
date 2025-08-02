import itertools
import math
import pathlib
import random
import pandas as pd
import numpy as np
from collections import Counter
from typing import Callable, Optional, Tuple
from Bio import Align, SeqIO
from Bio.SeqRecord import SeqRecord
import Bio.SeqRecord
from abc import ABC

from aldiscore.enums.enums import StringEnum

import traceback

_FEATURE_FLAG = "_is_feature"


def _feature(func):
    """Decorator flag for all functions that compute features."""
    func._is_feature = True
    return func


class BaseFeatureExtractor(ABC):
    """Base class for a feature extractor."""

    def __init__(self, sequences: list[SeqRecord]):
        self._sequences = sequences

    def compute(self, exclude: list = None):
        "Runs all functions with the @_feature decorator and concatenates the output."

        features_dict = {}
        feature_funcs = self._get_feature_funcs()
        if exclude is not None:
            feature_funcs = list(
                filter(lambda func: func not in exclude, feature_funcs)
            )
        for feature_func in feature_funcs:
            try:
                feature_dict = feature_func()
                assert isinstance(feature_dict, dict)
                features_dict.update(feature_dict)
            except Exception as e:
                raise ValueError(
                    f"Problem with feature {feature_func}", e, traceback.format_exc()
                )
        features = pd.DataFrame(features_dict, index=[0])  # Contains a single row
        return features

    def _get_feature_funcs(self):
        feature_funcs = []
        methods_list = [
            getattr(self, method)
            for method in self.__dir__()
            if callable(getattr(self, method))
        ]
        for attr in methods_list:
            if hasattr(attr, _FEATURE_FLAG):
                feature_funcs.append(attr)
        return feature_funcs

    @classmethod
    def descriptive_statistics(cls, series: list, name: str | StringEnum):
        """Computes a range of descriptive statistics on a series (min, max, mean, etc.)."""
        feat_dict = {}
        feat_dict["min_" + name] = min(series)
        feat_dict["max_" + name] = max(series)
        feat_dict["mean_" + name] = np.mean(series)
        feat_dict["median_" + name] = np.median(series)
        feat_dict["std_" + name] = np.std(series)
        # iqr = np.percentile(series, 75) - np.percentile(series, 25)
        # feature_dict["iqr_" + name] = iqr
        return feat_dict

    # # # # # features # # # # #


class AlDIFeatureExtractor(BaseFeatureExtractor):

    @_feature
    def _data_type(self) -> dict[str, list]:
        name = "data_type"
        feat = str(self._sequences.data_type)
        feat_dict = {name: feat}
        return feat_dict

    @_feature
    def _num_sequences(self) -> dict[str, list]:
        name = "n_sequences"
        feat = len(self._sequences.sequences)
        feat_dict = {name: feat}
        return feat_dict

    # @_feature
    # def _num_unique_sequences(self) -> dict[str, list]:
    #     name = "n_unique_sequences"
    #     feat = len(set(seq.seq for seq in self._dataset.sequences))
    #     feat_dict = {name: feat}
    #     return feat_dict

    @_feature
    def _sequence_length(self) -> dict[str, list]:
        name = "sequence_length"
        feat = self._sequences._sequence_lengths
        feat_dict = self.descriptive_statistics(feat, name)
        return feat_dict

    @_feature
    def _sequence_length_ratio(self) -> dict[str, list]:
        name = "sequence_length_ratio"
        feat = min(self._sequences._sequence_lengths) / max(
            self._sequences._sequence_lengths
        )

        feat_dict = {name: feat}
        return feat_dict

    @_feature
    def _lower_bound_gap_percentage(self) -> dict[str, list]:
        name = "lower_bound_gap_percentage"
        feat = 1 - np.mean(self._sequences._sequence_lengths) / max(
            self._sequences._sequence_lengths
        )

        feat_dict = {name: feat}
        return feat_dict

    @_feature
    def _sequence_length_taxa_ratio(self) -> dict[str, list]:
        name = "seq_length_taxa_ratio"
        feat = np.array(self._sequences._sequence_lengths) / len(
            self._sequences.sequences
        )
        feat_dict = self.descriptive_statistics(feat, name)
        return feat_dict

    @_feature
    def _sequence_shannon_entropy(self) -> dict[str, list]:
        """Computes the {min, max, mean ...} intra-sequence Shannon Entropy."""
        name = "shannon_entropy"
        feat = list(map(self._compute_sequence_entropy, self._sequences.sequences))
        feat_dict = self.descriptive_statistics(feat, name)
        return feat_dict

    @_feature
    def _pairwise_features(self) -> dict[str, list]:
        """Computes a bunch of features based on pairwise alignments of the unaligned sequences.
        - basic pairwise alignments: penalties = (1, 0, 0, 0)
        - advanced pairwise alignments: penalties = (2, -0.5, 0, -3) (Chowdhury and Garai, 2017)
        """
        penalty_settings = [(1, 0, 0, 0), (2, -0.5, 0, -3)]
        suffixes = ["_basic", "_advanced"]
        feat_dict = {}
        for penalties, suffix in zip(penalty_settings, suffixes):
            alignments = self._pairwise_alignments(penalties)

            name = "score_ratio" + suffix
            feat = self._alignment_score_ratio(alignments)
            feat_dict.update(self.descriptive_statistics(feat, name))

            name = "gap_ratio" + suffix
            feat = self._alignment_gap_ratio(alignments)
            feat_dict.update(self.descriptive_statistics(feat, name))

            name = "stretch_ratio" + suffix
            feat = self._alignment_stretch_ratio(alignments)
            feat_dict.update(self.descriptive_statistics(feat, name))

            name = "avg_gap_length" + suffix
            feat = self._average_gap_length(alignments)
            feat_dict.update(self.descriptive_statistics(feat, name))
        return feat_dict

    @_feature
    def _perc_seq_hash_hamming_distance(self) -> dict[str, list]:
        bit_suffix_lengths = [16]
        suffixes = ["_16bit"]
        feat_dict = {}
        for bit_suffix_length, suffix in zip(bit_suffix_lengths, suffixes):
            name = "perc_seq_hash_hamming" + suffix
            feat = self._compute_hamming_distance(bit_suffix_length)
            feat_dict.update(self.descriptive_statistics(feat, name))
        return feat_dict

    @_feature
    def _kmer_similarity(self) -> dict[str, list]:
        kmer_lengths = [5, 10]
        suffixes = ["_5", "_10"]
        feat_dict = {}
        for kmer_length, suffix in zip(kmer_lengths, suffixes):
            name = "kmer_similarity" + suffix
            feat = self._compute_kmer_similarity(kmer_length)
            feat_dict.update(self.descriptive_statistics(feat, name))
        return feat_dict

    # # # # # helper methods # # # # #

    def _compute_sequence_entropy(self, sequence: SeqIO.SeqRecord) -> float:
        char_counts = Counter(sequence.seq)
        sequence_length = len(sequence.seq)
        char_probabilities = [
            char_count / sequence_length for char_count in char_counts.values()
        ]
        entropy = -sum(p * math.log2(p) for p in char_probabilities if p > 0)
        return entropy

    def _pairwise_alignments(
        self,
        penalties: Tuple[float, float, float, float] = (1, 0, 0, 0),
        seed: int = 0,
    ) -> list[Bio.Align.Alignment]:
        """
        Returns pairwise alignments between all unique pairs of sequences in sequence_records.
        If the dataset contains more than 50 sequences, this method will only perform pairwise sequence alignment for
        a random sample of 1250 pairs.

        The pairwise alignment is scored using the given penalties.


        Parameters
        ----------
        penalties : Tuple[float, float, float, float], default=(1, 0, 0, 0)
            Penalty scores to use to find the best pairwise alignment.
            Expected are four values: match_score, mismatch_score, open_gap_score, extend_gap_score.
        seed : int, default=0
            Seed to initialize the random number generator with. This will only have an impact for datasets with
            more than 50 sequences, and it will influence the random sampling of pairs.
        threads : int, default=None
            How many threads to use for parallel computation of the pairwise alignments.
            Defaults to using all available system threads.

        Returns
        -------
        alignments : list[Bio.Align.Alignment]
            Returns a list of pairwise alignments as Bio.Align.Alignment objects.
        """

        aligner = Align.PairwiseAligner()
        (
            aligner.match_score,
            aligner.mismatch_score,
            aligner.open_gap_score,
            aligner.extend_gap_score,
        ) = penalties

        sequence_pairs = itertools.combinations(self._sequences.sequences, r=2)
        if len(self._sequences.sequences) > 50:
            random.seed(seed)
            sequence_pairs = random.sample(list(sequence_pairs), k=1250)

        alignments = []
        for seq_pair in sequence_pairs:
            alignments.append(aligner.align(*seq_pair)[0])

        return alignments

    def _alignment_score_ratio(self, alignments) -> list:
        """Score ratio = Alignment score scaled by the minimum sequence length"""
        score_ratios = []
        for alignment in alignments:
            s1, s2 = alignment.sequences
            score_ratio = alignment.score / min(len(s1), len(s2))
            score_ratios.append(score_ratio)
        return score_ratios

    def _alignment_gap_ratio(self, alignments) -> list:
        """Gap ratio = number of gaps divided by the total number of characters in the pairwise alignment"""
        gap_ratios = []
        for alignment in alignments:
            gap_ratio = alignment.counts().gaps / alignment.length * 2
            gap_ratios.append(gap_ratio)
        return gap_ratios

    def _alignment_stretch_ratio(self, alignments) -> list:
        """Stretch ratio = Increase in length of the longest unaligned sequence compared to the pairwise alignment"""
        stretch_ratios = []
        for alignment in alignments:
            s1, s2 = alignment.sequences
            stretch_ratio = max(len(s1), len(s2)) / alignment.length
            stretch_ratios.append(stretch_ratio)
        return stretch_ratios

    def _average_gap_length(self, alignments) -> list:
        """Average gap lengths."""
        avg_gap_lengths = []
        for alignment in alignments:
            s1, s2 = alignment.sequences
            avg_gap_lengths.append(self._seq_avg_gap_len(str(s1.seq)))
            avg_gap_lengths.append(self._seq_avg_gap_len(str(s2.seq)))
        return avg_gap_lengths

    def _seq_avg_gap_len(self, sequence: str):
        """Computes the average gap length by looping through the sequences."""
        sequence = sequence.strip("-")
        gap_lens = []
        seen_gap = False
        cur_len = 0
        for c in sequence:
            if c == "-":
                seen_gap = True
                cur_len += 1
            elif seen_gap:
                seen_gap = False
                gap_lens.append(cur_len)
                cur_len = 0

        if cur_len:
            gap_lens.append(cur_len)
        if gap_lens:
            return np.mean(gap_lens)
        else:
            return 0

    def _compute_hamming_distance(self, hash_size: Optional[int] = 16) -> list:
        """
        Computes the average perceptual hash hamming distance between all pairs of sequences in `sequences`.
        See the `sequence_perceptual_hash` function in `utils.py` for details on perceptual hashing.

        Parameters
        ----------
        hash_size : int, default=16
            Length of the hash to compute and compare. The default length is 16.

        Returns
        -------
        distances
            The distances between the perceptual hashes of all pairs of sequences.
        """
        distances = []
        sequences = SeqIO.parse(self._sequences.file_path, format="fasta")
        for seq1, seq2 in itertools.combinations(sequences, r=2):
            h1 = sequence_perceptual_hash(
                seq1.seq, self._sequences.data_type, hash_size
            )
            h2 = sequence_perceptual_hash(
                seq2.seq, self._sequences.data_type, hash_size
            )
            dist = normalized_hamming_dist(h1, h2)
            distances.append(dist)
        return distances

    def _compute_kmer_similarity(
        self, k: int, n_kmers: int = 1000, seed: int = 0
    ) -> list:
        if min(self._sequences._sequence_lengths) < k:
            return np.nan

        random.seed(seed)
        n_sequences = len(self._sequences.sequences)
        frequencies = []
        for _ in range(n_kmers):
            # 1. choose a random sequence
            reference_sequence = random.choice(self._sequences.sequences).seq
            # 2. choose a random k-mer of length k
            start_idx = random.randint(0, len(reference_sequence) - k)
            kmer = reference_sequence[start_idx : start_idx + k]
            # 3. Count how many other sequences contain this k-mer
            num_contains_kmer = len(
                [seq for seq in self._sequences.sequences if kmer in seq.seq]
            )
            frequencies.append((num_contains_kmer - 1) / (n_sequences - 1))

        return frequencies
