import itertools
import numpy as np
from abc import ABC
from aldiscore.scoring import utils
from aldiscore.scoring import encoding
from aldiscore.enums.enums import FeatureEnum as FE, PositionalEncodingEnum
from aldiscore.constants.constants import GAP_CONST, GAP_CHAR
from aldiscore.datastructures.ensemble import Ensemble
from aldiscore.datastructures.alignment import Alignment
from typing import Literal


class _Metric(ABC):
    enum = None
    name = None
    _cache = None
    _dtype = np.int32

    def __init__(self, cache: dict = None):
        self._cache = cache

    def compute(
        self,
        ensemble: Ensemble,
        cache: dict = {},
        format: Literal["flat", "matrix"] = "flat",
    ):
        scores = []
        self._cache = cache

        index_pairs = list(itertools.combinations(range(len(ensemble.alignments)), r=2))
        for idx_x, idx_y in index_pairs:
            score = self.compute_similarity(
                ensemble.alignments[idx_x],
                ensemble.alignments[idx_y],
            )
            scores.append(score)

        out = np.array(scores)
        if format == "matrix":
            out = utils.format_dist_mat(scores, ensemble)

        return out

    def compute_similarity(
        self,
        alignment_x: Alignment,
        alignment_y: Alignment,
    ) -> float:
        """
        Computes the pairwise distances of alignments in the ensemble.

        Parameters
        ----------
        - alignment_x: Alignment
            First alignment for the distance computation.
        - alignment_y: Alignment
            Second alignment for the distance computation.

        Returns
        -------
        -dist: float
            The computed distance
        """
        raise NotImplementedError()

    def _init_key_map(
        self,
        alignment_x: Alignment,
        alignment_y: Alignment,
        key_x=None,
        key_y=None,
    ):
        self._key_x = alignment_x.key if key_x is None else key_x
        self._key_y = alignment_y.key if key_y is None else key_y
        self._alignment_map = {self._key_x: alignment_x, self._key_y: alignment_y}

    def _get_metric_prerequisites(self, key):
        S = np.array(self._alignment_map[key].msa)
        Q = encoding.gapped_index_mapping(S, self._dtype)
        K = len(S)
        L_k_list = [len(arr) for arr in Q]
        L_max = max(L_k_list)
        return S, Q, K, L_k_list, L_max

    def _get_from_cache(self, key, compute_cachables: callable, **kwargs):
        use_cache = self._cache is not None
        cached_content = None
        if use_cache:
            if self.name not in self._cache:
                self._cache[self.name] = {}
            cached_content = self._cache[self.name].get(key, None)

        if cached_content is not None:
            out = cached_content
        else:
            out = compute_cachables(key, **kwargs)

        if use_cache and (cached_content is None):
            self._cache[self.name][key] = out
        return out

    def _everything_cached(self):
        return (
            (self._cache is not None)
            and (self.name in self._cache)
            and (self._key_x in self._cache[self.name])
            and (self._key_y in self._cache[self.name])
        )


class _HomologySetMetric(_Metric):
    def _avg_jaccard_coef(self, encoding_enum: PositionalEncodingEnum):
        # Used for d_SSP

        def from_scratch(key):
            S, Q, K, L_k_list, L_max = self._get_metric_prerequisites(key)
            A_code = encoding.encode_positions(S, encoding_enum, int_dtype=self._dtype)
            hcols_list, gap_mask_list = [], []
            for k in range(K):
                hcols_list.append(np.delete(A_code[:, Q[k]], k, axis=0))
                gap_mask_list.append(hcols_list[-1] != GAP_CONST)
            return hcols_list, gap_mask_list

        hcols_x_list, gap_mask_x_list = self._get_from_cache(self._key_x, from_scratch)
        hcols_y_list, gap_mask_y_list = self._get_from_cache(self._key_y, from_scratch)

        intersects = []
        unions = []
        for k in range(len(hcols_x_list)):
            seq_intersects = np.sum(
                (hcols_x_list[k] == hcols_y_list[k])
                & gap_mask_x_list[k]
                & gap_mask_y_list[k],
                axis=0,
            )
            # |AUB| = |A|+|B| - |A∩B|
            seq_unions = (
                np.sum(gap_mask_x_list[k], axis=0)
                + np.sum(gap_mask_y_list[k], axis=0)
                - seq_intersects
            )
            intersects.append(np.sum(seq_intersects))
            unions.append(np.sum(seq_unions))
        dist = 1 - np.sum(intersects) / np.sum(unions)
        return dist

    def _avg_hamming_dist(self, encoding_enum: PositionalEncodingEnum):
        # Used for d_seq, d_pos

        def from_scratch(key):
            S, Q, K, L_k_list, L_max = self._get_metric_prerequisites(key)
            A_code = encoding.encode_positions(S, encoding_enum, int_dtype=self._dtype)
            hcols_list = []
            for k in range(K):
                hcols_list.append(np.delete(A_code[:, Q[k]], k, axis=0))
            return hcols_list, K, L_k_list

        hcols_x_list, K, L_k_list = self._get_from_cache(self._key_x, from_scratch)
        hcols_y_list, K, L_k_list = self._get_from_cache(self._key_y, from_scratch)

        norm_hamming_dist = []
        for k in range(len(hcols_x_list)):
            seq_dists = np.sum(hcols_x_list[k] != hcols_y_list[k])
            norm_hamming_dist.append(seq_dists)

        num_total_sites = np.sum(L_k_list)
        dist = np.sum(norm_hamming_dist) / (num_total_sites * (K - 1))
        return dist


# # # # # Subclasses # # # # #


class SSPDistance(_HomologySetMetric):
    enum = FE.SSP_DIST
    name = str(enum)

    def compute_similarity(self, alignment_x: Alignment, alignment_y: Alignment):
        self._init_key_map(alignment_x, alignment_y)
        return self._avg_jaccard_coef(PositionalEncodingEnum.UNIFORM)


class DSeqDistance(_HomologySetMetric):
    enum = FE.D_SEQ_DIST
    name = str(enum)

    def compute_similarity(self, alignment_x: Alignment, alignment_y: Alignment):
        self._init_key_map(alignment_x, alignment_y)
        return self._avg_hamming_dist(PositionalEncodingEnum.SEQUENCE)


class DPosDistance(_HomologySetMetric):
    enum = FE.D_POS_DIST
    name = str(enum)

    def compute_similarity(self, alignment_x: Alignment, alignment_y: Alignment):
        self._init_key_map(alignment_x, alignment_y)
        return self._avg_hamming_dist(PositionalEncodingEnum.POSITION)


class PHashDistance(_Metric):
    enum = FE.PERC_HASH_HAMMING
    name = None

    def __init__(self, hash_size: int = 16, cache: dict = None):
        self.name = self.enum + f"_{hash_size}bit"
        self._hash_size = hash_size
        self._cache = cache

    def compute_similarity(self, alignment_x: Alignment, alignment_y: Alignment):
        self._init_key_map(alignment_x, alignment_y)

        def from_scratch(key, max_len: int):
            A_pad = self._pad_msa(np.array(self._alignment_map[key].msa), max_len)
            hash = utils.msa_perceptual_hash(
                A_pad, self._alignment_map[key].data_type, self._hash_size
            )
            return hash

        max_len = max(alignment_x.shape[1], alignment_y.shape[1])
        hash_x = self._get_from_cache(self._key_x, from_scratch, max_len=max_len)
        hash_y = self._get_from_cache(self._key_y, from_scratch, max_len=max_len)

        dist = utils.normalized_hamming_dist(hash_x, hash_y)
        return dist

    def _pad_msa(self, msa: np.ndarray, length: int) -> np.ndarray:
        cols_to_pad = length - msa.shape[1]
        if not cols_to_pad:
            return msa
        return np.concatenate(
            [msa, np.full((len(msa), cols_to_pad), fill_value=GAP_CHAR)], axis=1
        )


_CUSTOM_METRICS: list[_Metric] = [
    SSPDistance(),
    DSeqDistance(),
    DPosDistance(),
    PHashDistance(),
]
