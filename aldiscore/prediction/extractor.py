import math
from pathlib import Path
import datetime
import random
import pandas as pd
import numpy as np
from collections import Counter
from typing import Literal, Optional
from Bio.SeqRecord import SeqRecord
from abc import ABC
import itertools as it
from aldiscore import get_from_config
from aldiscore.enums.enums import StringEnum
from aldiscore.datastructures.utils import infer_data_type
from aldiscore.constants.constants import GAP_CHAR, GAP_CODE, STAT_SEP
import aldiscore.prediction.utils as utils
import traceback
import tempfile
import subprocess
from collections import defaultdict
from time import perf_counter
import parasail
from functools import partial

_FEATURE_FLAG = "_is_feature"


def _feature(func):
    """Decorator flag for all functions that compute features."""

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    wrapper._is_feature = True
    wrapper._name = func.__name__
    return wrapper


class BaseFeatureExtractor(ABC):
    """Base class for a feature extractor."""

    def __init__(
        self,
        sequences: list[SeqRecord],
        track_perf: bool = False,
    ):

        self._sequences = sequences
        self._cache = {}

        self._track_perf = track_perf
        if self._track_perf:
            self._perf_dict = {}
        else:
            self._perf_dict = None

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
                if self._track_perf:
                    t1 = perf_counter()
                    feature_dict = feature_func()
                    t2 = perf_counter()
                    # Wrapper has custom attribute _name
                    self._perf_dict[feature_func._name] = t2 - t1
                else:
                    feature_dict = feature_func()
                assert isinstance(feature_dict, dict)
                features_dict.update(feature_dict)
            except Exception:
                traceback.print_exc()
                raise ValueError(f"Problem with feature {feature_func}")

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

    @classmethod
    def descriptive_statistics(cls, series: list, name: str | StringEnum):
        """Computes a range of descriptive statistics on a series (min, max, mean, etc.)."""
        feat_dict = {}
        feat_dict["min" + STAT_SEP + name] = np.min(series)
        feat_dict["max" + STAT_SEP + name] = np.max(series)
        feat_dict["mean" + STAT_SEP + name] = np.mean(series)
        feat_dict["std" + STAT_SEP + name] = np.std(series)
        # thresholds = [1, 5, 10, 25, 40, 50, 60, 75, 90, 95, 99]
        thresholds = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
        # thresholds = [1, 5, 10, 25, 50, 75, 90, 95, 99]

        percentiles = np.percentile(series, thresholds)
        for t, p in zip(thresholds, percentiles):
            feat_dict[f"p{t}" + STAT_SEP + name] = p
        feat_dict["iqr" + STAT_SEP + name] = (
            feat_dict["p80" + STAT_SEP + name] - feat_dict["p20" + STAT_SEP + name]
        )
        return feat_dict

    # # # # # features # # # # #


class FeatureExtractor(BaseFeatureExtractor):
    _SEQ_ORD = "seq_ord"
    _SEQ_LEN = "seq_length"
    _CHAR_DIST = "char_dist"
    _PSA = "psa"
    _PSA_GROUPS = "psa_groups"
    _PSA_SCORES = "psa_scores"
    _PSA_INDEX_MAP = "psa_index_map"
    _DTYPE = "dtype"
    _INT_TYPE = "int_type"

    def __init__(
        self,
        sequences: list[SeqRecord],
        psa_config: dict = None,
        track_perf: bool = False,
        validate: Literal["warn", "error"] = "error",
    ):
        super().__init__(sequences, track_perf)

        self._validate_inputs(validate)
        config = self._init_psa_config()
        if psa_config is None:
            psa_config = {}
        config["DNA"].update(psa_config.get("DNA", {}))
        config["AA"].update(psa_config.get("AA", {}))
        self._psa_config = config

    # ----------------------------------------------
    # --------------- INIT HELPERS -----------------
    # ----------------------------------------------

    def _validate_inputs(self, validate: Literal["warn", "error"]):
        n = len(self._sequences)
        if n <= 2:
            msg = f"WARNING: Need at least 3 sequences, found {n}"
            if validate == "error":
                raise ValueError(msg)
            else:
                print(msg)

        k = min(len(seq) for seq in self._sequences)
        if k <= 1:
            msg = f"WARNING: Found sequence with length {k}"
            if validate == "error":
                raise ValueError(msg)
            else:
                print(msg)

    def _init_psa_config(self) -> dict[str, dict]:
        config = {}
        config["DNA"] = {"op": 5, "ep": 2, "matrix": parasail.dnafull}
        config["AA"] = {"op": 10, "ep": 1, "matrix": parasail.blosum62}
        config["MAX_COUNT"] = 1000
        config["GROUP_SIZE"] = 3
        return config

    # ------------------------------------------
    # --------- "HEAVY" INIT HELPERS -----------
    # ------------------------------------------

    # @_feature
    # # _init_cache needs to be called as a feature to support performance logs
    # def _init_cache(self) -> dict[str, str]:
    #     self._cache[self._SEQ_ORD] = [
    #         list(map(ord, str(seq_record.seq).upper()))
    #         for seq_record in self._sequences
    #     ]
    #     self._cache[self._DTYPE] = str(infer_data_type(self._sequences))
    #     self._cache[self._SEQ_LEN] = np.array([len(seq) for seq in self._sequences])

    #     will_overflow = self._cache[self._SEQ_LEN].max() >= 2**15
    #     self._cache[self._INT_TYPE] = np.int32 if will_overflow else np.int16

    #     self._cache[self._CHAR_DIST] = self._get_char_distributions()
    #     alignments, groupings, scores = self._get_pairwise_alignments()
    #     self._cache[self._PSA] = alignments
    #     self._cache[self._PSA_GROUPS] = groupings
    #     self._cache[self._PSA_SCORES] = scores
    #     self._cache[self._PSA_INDEX_MAP] = self._get_psa_index_map()
    #     return {}

    @_feature
    # _init_cache needs to be called as a feature to support performance logs
    def _init_basics(self) -> dict:
        self._cache[self._SEQ_ORD] = [
            list(map(ord, str(seq_record.seq).upper()))
            for seq_record in self._sequences
        ]
        self._cache[self._DTYPE] = str(infer_data_type(self._sequences))
        self._cache[self._SEQ_LEN] = np.array([len(seq) for seq in self._sequences])

        will_overflow = self._cache[self._SEQ_LEN].max() >= 2**15
        self._cache[self._INT_TYPE] = np.int32 if will_overflow else np.int16
        return {}

    @_feature
    def _init_char_dists(self) -> dict:
        self._cache[self._CHAR_DIST] = self._get_char_distributions()
        return {}

    @_feature
    def _init_psa(self) -> dict:
        alignments, groupings, scores = self._get_pairwise_alignments()
        self._cache[self._PSA] = alignments
        self._cache[self._PSA_GROUPS] = groupings
        self._cache[self._PSA_SCORES] = scores
        return {}

    @_feature
    def _init_psa_index_map(self) -> dict:
        self._cache[self._PSA_INDEX_MAP] = self._get_psa_index_map()
        return {}

    # ------------------------------------------
    # --------------- FEATURES -----------------
    # ------------------------------------------

    @_feature
    def _data_type(self) -> dict[str, bool]:
        name = "is_dna"
        feat = self._cache[self._DTYPE].upper() == "DNA"
        feat_dict = {name: feat}
        return feat_dict

    @_feature
    def _num_sequences(self) -> dict[str, int]:
        name = "num_seqs"
        feat = len(self._sequences)
        feat_dict = {name: feat}
        return feat_dict

    # @_feature
    # def _num_unique_sequences(self) -> dict[str, float]:
    #     name = "n_unique_sequences"
    #     feat = len(set(seq.seq for seq in self._dataset.sequences))
    #     feat_dict = {name: feat}
    #     return feat_dict

    @_feature
    def _sequence_length(self) -> dict[str, float]:
        name = "seq_length"
        feat = self._get_cached(self._SEQ_LEN)
        feat_dict = self.descriptive_statistics(feat, name)
        return feat_dict

    @_feature
    def _sequence_length_ratio(self) -> dict[str, float]:
        name = "seq_length_ratio"
        seq_lengths = self._get_cached(self._SEQ_LEN)
        feat = min(seq_lengths) / max(seq_lengths)
        feat_dict = {name: feat}
        return feat_dict

    @_feature
    def _lower_bound_gap_percentage(self) -> dict[str, float]:
        name = "lower_bound_gap_percentage"
        seq_lengths = self._get_cached(self._SEQ_LEN)
        feat = 1 - np.mean(seq_lengths) / max(seq_lengths)
        feat_dict = {name: feat}
        return feat_dict

    # @_feature
    # def _sequence_length_taxa_ratio(self) -> dict[str, float]:
    #     name = "seq_length_taxa_ratio"
    #     seq_lengths = [len(seq) for seq in self._sequences]
    #     feat = seq_lengths / len(self._sequences)
    #     feat_dict = self.descriptive_statistics(feat, name)
    #     return feat_dict

    # # TODO: Included in FRST randomness features!
    # @_feature
    # def _sequence_entropy(self) -> dict[str, float]:
    #     """Computes the {min, max, mean ...} intra-sequence Shannon Entropy."""
    #     eps = 1e-8
    #     name = "entropy"
    #     dists = self._get_cached(self._CHAR_DIST).clip(eps)
    #     dists = dists.clip(eps)
    #     feat = -(dists * np.log2(dists)).sum(axis=1) / (-np.log2(1 / len(dists)))
    #     feat_dict = self.descriptive_statistics(feat, name)
    #     return feat_dict

    # @_feature
    # def _sequence_cross_entropy(self) -> dict[str, float]:
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
    def _char_js_divergence(self) -> dict[str, float]:
        """Computes the {min, max, mean ...} pairwise Jensen-Shannon Divergence."""
        eps = 1e-8
        name = "js_char"
        # Re-normalized in utils.js_divergence
        dist = self._get_cached(self._CHAR_DIST).clip(eps)
        comb_idxs = np.array(list(it.combinations(np.arange(len(dist)), r=2))).T
        js = utils.js_divergence(dist[comb_idxs[0]], dist[comb_idxs[1]], axis=1)
        feat_dict = self.descriptive_statistics(js, name)
        return feat_dict

    @_feature
    def _homopolymer_js_divergence(self) -> dict[str, float]:
        """Computes the {min, max, mean ...} pairwise Jensen-Shannon Divergence."""
        eps = 1e-8
        name = "js_hpoly"
        tags = ["count", "len"]
        dists = utils.repeat_distributions(self._get_cached(self._SEQ_ORD))
        feat_dict = {}
        for tag, dist in zip(tags, dists):
            # Re-normalized in utils.js_divergence
            dist = dist.clip(eps)
            comb_idxs = np.array(list(it.combinations(np.arange(len(dist)), r=2))).T
            js = utils.js_divergence(dist[comb_idxs[0]], dist[comb_idxs[1]], axis=1)
            feat_dict.update(self.descriptive_statistics(js, name + "_" + tag))
        return feat_dict

    @_feature
    def _ent_randomness(self) -> dict[str, float]:
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
                keys = [name + "_" + key.strip().lower() for key in lines[0]]
                # Drop first position (csv index)
                for key, val in zip(keys[1:], lines[1][1:]):
                    if key.endswith("monte-carlo-pi"):
                        continue
                    elif key.endswith("chi-square"):
                        # Correlates almost perfectly with seq_length
                        eps = 1  # Instability only with unrealistically short sequences
                        key = name + "_" + "inv-chi-square"
                        val = len(seq) / (float(val.strip()) + eps)
                    else:
                        val = float(val.strip())
                    feats[key].append(val)

        del feats[name + "_file-bytes"]  # Redundant, same as sequence length
        feat_dict = {}
        for key, feat in feats.items():
            feat_dict.update(self.descriptive_statistics(feat, key))
        return feat_dict

    @_feature
    def _transitive_consistency(self) -> dict[str, float]:
        """Computes measures of transitive consistency on the PSA groups."""
        name = "tc_base"
        similarity_func = lambda x, y: np.sum(x == y) / len(x)
        scores = self._compute_consistency(name, similarity_func)

        feat_dict = {}
        for tag in scores:
            feat_dict.update(self.descriptive_statistics(scores[tag], tag))
        return feat_dict

    # @_feature
    # def _transitive_consistency_dist_scaled(self) -> dict[str, float]:
    #     """Computes measures of transitive consistency on the PSA groups."""
    #     name = "tc_dist_scaled"
    #     similarity_func = lambda x, y: np.linalg.norm(x - y)
    #     scores = self._compute_consistency(name, similarity_func)
    #     feat_dict = {}
    #     max_len = max(self._get_cached(self._SEQ_LEN))
    #     for tag in scores:
    #         scaled = np.array(scores[tag]) / max_len
    #         feat_dict.update(self.descriptive_statistics(scaled, tag))
    #     return feat_dict

    @_feature
    def _psa_basic_features(self) -> dict[str, float]:
        """Computes a bunch of features based on pairwise alignments of the unaligned sequences."""
        score_dict = defaultdict(list)
        al_scores = self._get_cached(self._PSA_SCORES)
        func_map = {
            "psa_score_ratio": self._alignment_score_ratio,
            "psa_gap_ratio": self._alignment_gap_ratio,
            "psa_stretch_ratio": self._alignment_stretch_ratio,
        }
        idx_cache = set()
        for name, func in func_map.items():
            for idx_pair in al_scores:
                if idx_pair not in idx_cache:
                    idx_cache.add(idx_pair)
                    score_dict[name].append(func(idx_pair))
        feat_dict = {}
        for name, vals in score_dict.items():
            feat_dict.update(self.descriptive_statistics(vals, name))
        return feat_dict

    @_feature
    def _psa_gap_length(self) -> dict[str, float]:
        """Computes features based on gap lengths."""
        name = "psa_gap_len"
        score_dict = defaultdict(list)
        # psa_scores only used for alignment index pairs here
        al_scores = self._get_cached(self._PSA_SCORES)
        idx_cache = set()
        for idx_pair in al_scores:
            if idx_pair not in idx_cache:
                idx_cache.add(idx_pair)
                gap_len_arr = self._gap_lengths(idx_pair, name)
                gap_ends = gap_len_arr > 0
                has_gaps = gap_ends.any(axis=1)

                gap_lens = gap_len_arr[gap_ends] if has_gaps.any() else np.array([0])

                score_dict[name + "_mean"].append(gap_lens.mean())
                score_dict[name + "_p50"].append(np.percentile(gap_lens, 50))
                score_dict[name + "_std"].append(gap_lens.std())

                iqr = np.percentile(gap_lens, 75) - np.percentile(gap_lens, 25)
                score_dict[name + "_iqr"].append(iqr)

                seq_means = np.zeros_like(has_gaps)
                if has_gaps[0]:
                    seq_means[0] = gap_len_arr[0, gap_ends[0]].mean()
                if has_gaps[1]:
                    seq_means[1] = gap_len_arr[1, gap_ends[1]].mean()
                diff = np.std(seq_means, ddof=1)
                if diff > 0:
                    diff /= seq_means.mean()
                score_dict[name + "_logdiff"].append(np.log2(diff + 1))

        feat_dict = {}
        for tag, vals in score_dict.items():
            feat_dict.update(self.descriptive_statistics(vals, tag))
        return feat_dict

    @_feature
    def _kmer_similarity(self) -> dict[str, float]:
        name = "mer"
        Ks = [3, 5, 7, 9]
        eps = 1e-8
        feat_dict = {}
        seqs = self._get_cached(self._SEQ_ORD)
        for k in Ks:
            count_table = defaultdict(
                partial(np.zeros, shape=len(self._sequences), dtype=np.int32)
            )
            for i, seq in enumerate(seqs):
                n = len(seq)
                if n < k:
                    continue
                for kmer in (tuple(seq[i : i + k]) for i in range(n - k + 1)):
                    count_table[hash(kmer)][i] += 1

            dist = np.stack(list(count_table.values()), axis=1)
            # If n < k (very short sequences), filter out
            dist = dist[dist.sum(axis=1) != 0]
            # Re-normalized in utils.js_divergence and utils.shannon_entropy
            dist = (dist / dist.sum(axis=1, keepdims=True)).clip(eps)
            comb_idxs = np.array(list(it.combinations(np.arange(len(dist)), r=2))).T
            js = utils.js_divergence(dist[comb_idxs[0]], dist[comb_idxs[1]], axis=1)
            feat_dict.update(self.descriptive_statistics(js, str(k) + name + "_js"))

            entros = utils.shannon_entropy(dist, axis=1)
            feat_dict.update(
                self.descriptive_statistics(entros, str(k) + name + "_ent")
            )

        return feat_dict

    # ----------------------------------------------
    # --------------- MISC HELPERS -----------------
    # ----------------------------------------------

    # def _compute_sequence_entropy(self, sequence: SeqIO.SeqRecord) -> float:
    #     dists = self._get_cached(self._CHAR_DIST)

    #     char_counts = Counter(sequence.seq)
    #     sequence_length = len(sequence.seq)
    #     char_probabilities = [
    #         char_count / sequence_length for char_count in char_counts.values()
    #     ]
    #     entropy = -sum(p * math.log2(p) for p in char_probabilities if p > 0)
    #     return entropy

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

    def _get_pairwise_alignments(self):
        datatype = self._cache[self._DTYPE]
        al_mat = self._psa_config[datatype]["matrix"]
        op = self._psa_config[datatype]["op"]
        ep = self._psa_config[datatype]["ep"]

        # If only 3 sequences, group_size>3 won't work
        # Less than 3 sequences should not be processed anyway
        group_size = max(min(self._psa_config["GROUP_SIZE"], len(self._sequences)), 3)
        psas_per_group = group_size * (group_size - 1) // 2
        max_num_groups = self._psa_config["MAX_COUNT"] // psas_per_group
        n = len(self._sequences)
        int_type = self._cache[self._INT_TYPE]
        psas = defaultdict(dict)
        scores = {}
        groupings = []

        # Compute sets of pairwise alignments
        for seq_group in utils.sample_index_tuples(n, r=group_size, k=max_num_groups):
            groupings.append(seq_group)
            for seq_pair in it.combinations(seq_group, r=2):
                idx_q, idx_r = seq_pair
                if (idx_q in psas) and (idx_r in psas[idx_q]):
                    # Already processed this alignment in another group
                    continue
                al = parasail.nw_trace_scan(
                    str(self._sequences[idx_q].seq).upper(),
                    str(self._sequences[idx_r].seq).upper(),
                    # TODO: Try random variations of penalties
                    # open=int(op + 0.25 * op * (random.random() * 2 - 1)),
                    open=op,
                    extend=ep,
                    matrix=al_mat,
                )
                psas[idx_q][idx_r] = np.array(
                    list(map(ord, al.traceback.query)), dtype=int_type
                )
                psas[idx_r][idx_q] = np.array(
                    list(map(ord, al.traceback.ref)), dtype=int_type
                )
                # Tuples are sorted, hence no need for checks here
                scores[seq_pair] = al.score
        return psas, groupings, scores

    def _get_psa_index_map(self) -> dict:
        int_type = self._cache[self._INT_TYPE]
        GAP_ORD = int_type(ord(GAP_CHAR))
        index_map = defaultdict(dict)
        for group in self._cache[self._PSA_GROUPS]:
            for idx_a, idx_b in it.product(group, group):
                duplicate_idx = idx_a == idx_b
                done = (idx_a in index_map) and (idx_b in index_map[idx_a])
                if duplicate_idx or done:
                    continue
                al_a = self._cache[self._PSA][idx_a][idx_b]
                al_b = self._cache[self._PSA][idx_b][idx_a]

                # vector with positional indices in b with respect to those in a
                positions = (np.cumsum(al_b != GAP_ORD) - 1).astype(int_type)
                positions[al_b == GAP_ORD] = GAP_CODE
                positions = positions[al_a != GAP_ORD]
                index_map[idx_a][idx_b] = positions
        return index_map

    def _compute_consistency(self, name: str, similarity_func) -> dict:
        groups = self._get_cached(self._PSA_GROUPS)
        index_map = self._get_cached(self._PSA_INDEX_MAP)
        score_dict = defaultdict(list)
        # Loop over groups of r sequences (usually r=3)
        for group in groups:
            group_scores = []
            # Loop over all pairwise combinations (with replacement)
            for idx_pair in it.product(group, group):
                idx_a, idx_c = idx_pair
                if idx_a == idx_c:
                    continue
                pos_ac = index_map[idx_a][idx_c]
                others = set(group).difference(idx_pair)
                # Loop over all remaining r-2 indices (with r=3 only 1 remaining)
                for idx_b in others:
                    pos_ab = index_map[idx_a][idx_b]
                    pos_bc = index_map[idx_b][idx_c]
                    mask = (pos_ab != GAP_CODE) & (pos_ac != GAP_CODE)
                    if np.sum(mask) > 0:
                        # nums.append(np.sum(pos_ac[mask] == pos_bc[pos_ab[mask]]))
                        group_scores.append(
                            similarity_func(pos_ac[mask], pos_bc[pos_ab[mask]])
                        )
            # Compute summary statistics (6 consistency values per group if r=3)
            group_scores = np.array(group_scores)
            score_dict[name + "_min"].append(group_scores.min())
            score_dict[name + "_mean"].append(group_scores.mean())
            score_dict[name + "_std"].append(group_scores.std())
            score_dict[name + "_max"].append(group_scores.max())
            score_dict[name + "_p50"].append(np.percentile(group_scores, 50))
            # for p in [1, 5, 10, 25, 75, 90, 95]:
            #     scores[name + f"_p{p}"].append(np.percentile(group_scores, p))
            # scores[name + "_iqr"] = scores[name + "p75"] - scores[name + "p25"]

        return score_dict

    def _alignment_score_ratio(self, idx_pair: tuple[int]) -> float:
        """Score ratio = Alignment score scaled by the minimum sequence length"""
        score = self._cache[self._PSA_SCORES][idx_pair]
        min_len = self._cache[self._SEQ_LEN][list(idx_pair)].min()
        return score / min_len

    def _alignment_gap_ratio(self, idx_pair: tuple[int]) -> float:
        """Gap ratio = number of gaps divided by the total number of characters in the pairwise alignment"""
        al_a = self._cache[self._PSA][idx_pair[1]][idx_pair[0]]
        al_b = self._cache[self._PSA][idx_pair[0]][idx_pair[1]]
        gap_ord = self._cache[self._INT_TYPE](ord(GAP_CHAR))
        num_gaps = np.sum(al_a == gap_ord) + np.sum(al_b == gap_ord)
        return num_gaps / (2 * len(al_a))

    def _alignment_stretch_ratio(self, idx_pair: tuple[int]) -> float:
        """Stretch ratio = Increase in length of the longest unaligned sequence compared to the pairwise alignment"""
        max_len = self._cache[self._SEQ_LEN][list(idx_pair)].max()
        return max_len / len(self._cache[self._PSA][idx_pair[1]][idx_pair[0]])

    def _gap_lengths(self, idx_pair: tuple[int], name: str) -> np.ndarray:
        """Returns gap length at end of every gap region (else 0)."""
        gap_ord = self._cache[self._INT_TYPE](ord(GAP_CHAR))
        al = np.stack(
            [
                self._cache[self._PSA][idx_pair[1]][idx_pair[0]],
                self._cache[self._PSA][idx_pair[0]][idx_pair[1]],
            ],
            dtype=self._cache[self._INT_TYPE],
        )
        gap_length_arr = utils.compute_gap_lengths(al, gap_ord)
        return gap_length_arr
