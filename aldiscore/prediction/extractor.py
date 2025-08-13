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
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
import itertools
from aldiscore import get_from_config
from aldiscore.enums.enums import StringEnum
from aldiscore.datastructures.utils import infer_data_type
from aldiscore.enums.enums import DataTypeEnum
import traceback
import tempfile
import subprocess
from collections import defaultdict
from time import time
import parasail


_FEATURE_FLAG = "_is_feature"
_LOG_TIME = False


def _feature(func):
    """Decorator flag for all functions that compute features."""

    def wrapper(*args, **kwargs):
        if _LOG_TIME:
            t1 = time()
            result = func(*args, **kwargs)
            print(f"{func.__name__}: {time() - t1:.6f}")
        else:
            result = func(*args, **kwargs)
        return result

    wrapper._is_feature = True
    return wrapper


class BaseFeatureExtractor(ABC):
    """Base class for a feature extractor."""

    def __init__(self, sequences: list[SeqRecord]):
        self._sequences = sequences
        self._cache = {}

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

    def _get_cached(self, name: str):
        return self._cache.get(name)

    def _set_cached(self, name: str, val):
        self._cache[name] = val

    @classmethod
    def descriptive_statistics(cls, series: list, name: str | StringEnum):
        """Computes a range of descriptive statistics on a series (min, max, mean, etc.)."""
        feat_dict = {}
        feat_dict["min:" + name] = np.min(series)
        feat_dict["max:" + name] = np.max(series)
        feat_dict["mean:" + name] = np.mean(series)
        feat_dict["std:" + name] = np.std(series)
        thresholds = [5, 10, 25, 50, 75, 90, 95]
        percentiles = np.percentile(series, thresholds)
        for t, p in zip(thresholds, percentiles):
            feat_dict[f"p{t}:" + name] = p
        # feat_dict["median_" + name] = np.median(series)
        # iqr = np.percentile(series, 75) - np.percentile(series, 25)
        # feature_dict["iqr_" + name] = iqr
        return feat_dict

    # # # # # features # # # # #


class FeatureExtractor(BaseFeatureExtractor):
    _SEQ_LEN = "seq_length"
    _CHAR_DIST = "char_dist"
    _PSA = "psa"
    _DTYPE = "dtype"

    @_feature
    def _init_cache(self) -> dict[str, str]:
        self._cache[self._SEQ_LEN] = [len(seq) for seq in self._sequences]
        self._cache[self._CHAR_DIST] = self._get_char_distributions()
        self._cache[self._PSA] = self._get_pairwise_alignments()
        self._cache[self._DTYPE] = str(infer_data_type(self._sequences))
        return {}

    @_feature
    def _data_type(self) -> dict[str, list]:
        name = "is_dna"
        feat = self._cache[self._DTYPE].upper() == "DNA"
        feat_dict = {name: feat}
        return feat_dict

    @_feature
    def _num_sequences(self) -> dict[str, list]:
        name = "num_seqs"
        feat = len(self._sequences)
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
        name = "seq_length"
        feat = self._cache[self._SEQ_LEN]
        feat_dict = self.descriptive_statistics(feat, name)
        return feat_dict

    # @_feature
    # def _sequence_length_ratio(self) -> dict[str, list]:
    #     name = "seq_length_ratio"
    #     seq_lengths = [len(seq) for seq in self._sequences]
    #     feat = min(seq_lengths) / max(seq_lengths)
    #     feat_dict = {name: feat}
    #     return feat_dict

    @_feature
    # TODO: probably redundant
    def _lower_bound_gap_percentage(self) -> dict[str, list]:
        name = "lower_bound_gap_percentage"
        seq_lengths = self._get_cached(self._SEQ_LEN)
        feat = 1 - np.mean(seq_lengths) / max(seq_lengths)
        feat_dict = {name: feat}
        return feat_dict

    # @_feature
    # def _sequence_length_taxa_ratio(self) -> dict[str, list]:
    #     name = "seq_length_taxa_ratio"
    #     seq_lengths = [len(seq) for seq in self._sequences]
    #     feat = seq_lengths / len(self._sequences)
    #     feat_dict = self.descriptive_statistics(feat, name)
    #     return feat_dict

    # # TODO: Included in FRST randomness features!
    # @_feature
    # def _sequence_entropy(self) -> dict[str, list]:
    #     """Computes the {min, max, mean ...} intra-sequence Shannon Entropy."""
    #     eps = 1e-8
    #     name = "entropy"
    #     dists = self._get_cached(self._CHAR_DIST).clip(eps)
    #     dists = dists.clip(eps)
    #     feat = -(dists * np.log2(dists)).sum(axis=1) / (-np.log2(1 / len(dists)))
    #     feat_dict = self.descriptive_statistics(feat, name)
    #     return feat_dict

    # @_feature
    # def _sequence_cross_entropy(self) -> dict[str, list]:
    #     """Computes the {min, max, mean ...} pairwise Cross-Entropy."""
    #     eps = 1e-8
    #     name = "cross_entropy"
    #     dists = self._get_cached("distributions")
    #     # dists = self._get_char_distributions()  # TODO remove after testing!
    #     dists = dists.clip(eps)  # k x v --> k x k
    #     entros = -np.einsum("iv,jv -> ij", dists, np.log2(dists))
    #     v = dists.shape[1]
    #     entros = (entros + entros.T) / (2 * -np.log2(1 / v))
    #     feat = entros[np.tril_indices_from(entros, k=-1)]
    #     feat_dict = self.descriptive_statistics(feat, name)
    #     return feat_dict

    @_feature
    def _sequence_js_divergence(self) -> dict[str, list]:
        """Computes the {min, max, mean ...} pairwise Jensen-Shannon Divergence."""
        eps = 1e-8
        name = "js_divergence"
        dists = self._get_cached(self._CHAR_DIST).clip(eps)
        comb_idxs = np.array(list(itertools.combinations(np.arange(len(dists)), r=2))).T
        # js = jensenshannon(dists[comb_idxs[0]], dists[comb_idxs[1]], axis=1)
        js = self._jensenshannon(dists[comb_idxs[0]], dists[comb_idxs[1]], axis=1)
        feat_dict = self.descriptive_statistics(js, name)
        return feat_dict

    @_feature
    def _get_ent_features(self):
        name = "frst"
        ent_path = get_from_config("tools", "ent")
        feats = defaultdict(list)
        with tempfile.NamedTemporaryFile() as tmpfile:
            for seq in self._sequences:
                with open(tmpfile.name, "wb") as file:
                    file.write(str(seq.seq).encode("utf-8"))
                cmd = [ent_path, "-t", tmpfile.name]
                out = subprocess.run(cmd, capture_output=True).stdout.decode("utf-8")
                lines = [line.split(",") for line in out.splitlines()]
                keys = [name + "_" + key.lower() for key in lines[0]]
                # Drop first position (csv index)
                for key, val in zip(keys[1:], lines[1][1:]):
                    feats[key].append(float(val))
        del feats[name + "_file-bytes"]  # Redundant, same as sequence length
        feat_dict = {}
        for name, feat in feats.items():
            feat_dict.update(self.descriptive_statistics(feat, name))
        return feat_dict

    # @_feature
    # def _pairwise_features(self) -> dict[str, list]:
    #     """Computes a bunch of features based on pairwise alignments of the unaligned sequences.
    #     - basic pairwise alignments: penalties = (1, 0, 0, 0)
    #     - advanced pairwise alignments: penalties = (2, -0.5, 0, -3) (Chowdhury and Garai, 2017)
    #     """
    #     penalty_settings = [(1, 0, 0, 0), (2, -0.5, 0, -3)]
    #     suffixes = ["_basic", "_advanced"]
    #     feat_dict = {}
    #     for penalties, suffix in zip(penalty_settings, suffixes):
    #         alignments = self._pairwise_alignments(penalties)

    #         name = "score_ratio" + suffix
    #         feat = self._alignment_score_ratio(alignments)
    #         feat_dict.update(self.descriptive_statistics(feat, name))

    #         name = "gap_ratio" + suffix
    #         feat = self._alignment_gap_ratio(alignments)
    #         feat_dict.update(self.descriptive_statistics(feat, name))

    #         name = "stretch_ratio" + suffix
    #         feat = self._alignment_stretch_ratio(alignments)
    #         feat_dict.update(self.descriptive_statistics(feat, name))

    #         name = "avg_gap_length" + suffix
    #         feat = self._average_gap_length(alignments)
    #         feat_dict.update(self.descriptive_statistics(feat, name))
    #     return feat_dict

    # @_feature
    # def _perc_seq_hash_hamming_distance(self) -> dict[str, list]:
    #     bit_suffix_lengths = [16]
    #     suffixes = ["_16bit"]
    #     feat_dict = {}
    #     for bit_suffix_length, suffix in zip(bit_suffix_lengths, suffixes):
    #         name = "perc_seq_hash_hamming" + suffix
    #         feat = self._compute_hamming_distance(bit_suffix_length)
    #         feat_dict.update(self.descriptive_statistics(feat, name))
    #     return feat_dict

    # @_feature
    # def _kmer_similarity(self) -> dict[str, list]:
    #     kmer_lengths = [5, 10]
    #     suffixes = ["_5", "_10"]
    #     feat_dict = {}
    #     for kmer_length, suffix in zip(kmer_lengths, suffixes):
    #         name = "kmer_similarity" + suffix
    #         feat = self._compute_kmer_similarity(kmer_length)
    #         feat_dict.update(self.descriptive_statistics(feat, name))
    #     return feat_dict

    # # # # # helper methods # # # # #

    def _compute_sequence_entropy(self, sequence: SeqIO.SeqRecord) -> float:
        dists = self._get_char_distributions()
        if self._get_cached("char_dists") is None:
            self._set_cached("char_dists", dists)

        char_counts = Counter(sequence.seq)
        sequence_length = len(sequence.seq)
        char_probabilities = [
            char_count / sequence_length for char_count in char_counts.values()
        ]
        entropy = -sum(p * math.log2(p) for p in char_probabilities if p > 0)
        return entropy

    def _get_char_distributions(self):
        """
        Given a list of Bio.SeqRecord objects,
        return a NumPy array of shape (n_seqs, vocab_size),
        where each row is the symbol distribution for a sequence.
        """
        # Extract all unique symbols across all sequences
        all_symbols = sorted(set("".join(str(rec.seq) for rec in self._sequences)))
        symbol_to_idx = {symbol: i for i, symbol in enumerate(all_symbols)}
        vocab_size = len(all_symbols)
        n_seqs = len(self._sequences)

        # Initialize array to hold distributions
        distributions = np.zeros((n_seqs, vocab_size), dtype=np.float32)

        # Fill in frequency counts for each sequence
        for i, rec in enumerate(self._sequences):
            seq = str(rec.seq)
            for symbol, count in Counter(rec.seq).items():
                distributions[i, symbol_to_idx[symbol]] = count

            # Normalize to get probabilities
            seq_len = len(seq)
            if seq_len > 0:
                distributions[i] /= seq_len

        return distributions

    def _jensenshannon(self, p, q, axis):
        p = p / np.sum(p, axis=axis, keepdims=True)
        q = q / np.sum(q, axis=axis, keepdims=True)
        m = (p + q) / 2
        left = np.sum(p * np.log(p / m), axis=1).clip(min=0)
        right = np.sum(q * np.log(q / m), axis=1).clip(min=0)
        js = np.sqrt((left + right) / 2)
        return js

    def _get_pairwise_alignments(self):
        OP, EP = (4, 2)
        AL_MAT = parasail.dnafull
        MAX_NUM_PSA = 1000

        # Compute sets of pairwise alignments
        seq_pairs = list(itertools.combinations(range(len(self._sequences)), r=2))
        if len(seq_pairs) > MAX_NUM_PSA:
            seq_pairs = random.sample(seq_pairs, k=MAX_NUM_PSA)

        alignments = {}
        for seq_pair in seq_pairs:
            al = parasail.nw_trace_striped_16(
                str(self._sequences[seq_pair[0]].seq),
                str(self._sequences[seq_pair[1]].seq),
                OP,
                EP,
                AL_MAT,
            )
            al_arr = np.array(
                [
                    [*map(ord, al.traceback.query.upper())],
                    [*map(ord, al.traceback.ref.upper())],
                ]
            )
            alignments[seq_pair].append(al_arr)
        return alignments

    def _pairwise_alignments_old(
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

    # def _compute_hamming_distance(self, hash_size: Optional[int] = 16) -> list:
    #     """
    #     Computes the average perceptual hash hamming distance between all pairs of sequences in `sequences`.
    #     See the `sequence_perceptual_hash` function in `utils.py` for details on perceptual hashing.

    #     Parameters
    #     ----------
    #     hash_size : int, default=16
    #         Length of the hash to compute and compare. The default length is 16.

    #     Returns
    #     -------
    #     distances
    #         The distances between the perceptual hashes of all pairs of sequences.
    #     """
    #     distances = []
    #     sequences = SeqIO.parse(self._sequences.file_path, format="fasta")
    #     for seq1, seq2 in itertools.combinations(sequences, r=2):
    #         h1 = sequence_perceptual_hash(
    #             seq1.seq, self._sequences.data_type, hash_size
    #         )
    #         h2 = sequence_perceptual_hash(
    #             seq2.seq, self._sequences.data_type, hash_size
    #         )
    #         dist = normalized_hamming_dist(h1, h2)
    #         distances.append(dist)
    #     return distances

    # def _compute_kmer_similarity(
    #     self, k: int, n_kmers: int = 1000, seed: int = 0
    # ) -> list:
    #     if min(self._sequences._sequence_lengths) < k:
    #         return np.nan

    #     random.seed(seed)
    #     n_sequences = len(self._sequences.sequences)
    #     frequencies = []
    #     for _ in range(n_kmers):
    #         # 1. choose a random sequence
    #         reference_sequence = random.choice(self._sequences.sequences).seq
    #         # 2. choose a random k-mer of length k
    #         start_idx = random.randint(0, len(reference_sequence) - k)
    #         kmer = reference_sequence[start_idx : start_idx + k]
    #         # 3. Count how many other sequences contain this k-mer
    #         num_contains_kmer = len(
    #             [seq for seq in self._sequences.sequences if kmer in seq.seq]
    #         )
    #         frequencies.append((num_contains_kmer - 1) / (n_sequences - 1))

    #     return frequencies
