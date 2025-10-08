"""
Implements pairwise distance metrics between alignments of the same dataset.

Provides metric classes for computing distances between alignments, with optional caching
to avoid redundant computation when evaluating similarity matrices.
"""

import itertools
import numpy as np
from abc import ABC
from aldiscore.scoring import utils
from aldiscore.scoring import encoding
from aldiscore.enums.enums import MethodEnum as ME, PositionalEncodingEnum
from aldiscore.constants.constants import GAP_CODE, GAP_CHAR
from aldiscore.datastructures.ensemble import Ensemble
from aldiscore.datastructures.alignment import Alignment
from typing import Literal


class _Metric(ABC):
    """
    Abstract base class for pairwise alignment distance metrics.

    Provides a caching mechanism and a standard interface for computing
    pairwise distance scores between alignments in an ensemble.

    Attributes
    ----------
    enum : MethodEnum
        Enum identifier for the metric.
    name : str
        Name of the metric.
    _format : {"scalar", "flat", "matrix"}, default="scalar"
        Output format: scalar, flat list, or square matrix.
    _cache : dict or None
        Cache for storing intermediate results.
    """

    def __init__(
        self,
        format: Literal["scalar", "flat", "matrix"] = "scalar",
        cache: dict = None,
    ):
        """
        Initialize the metric with an optional cache.

        Parameters
        ----------
        format : {"scalar", "flat", "matrix"}, default="scalar"
            Output format: scalar, flat list, or square matrix.
        cache : dict, optional
            Dictionary for caching intermediate results.
            Caching is applied by default. The parameter allows for re-using an existing cache.
        """
        super().__init__()
        self._format = format
        self._cache = {} if cache is None else cache

    def compute(
        self,
        ensemble: Ensemble,
        reference: Alignment = None,
    ):
        """
        Compute pairwise distance scores for all pairs in an ensemble.

        Parameters
        ----------
        ensemble : Ensemble
            Ensemble of alignments to compare.
        reference : Alignment, optional
            If provided, computes average distance with respect to reference.

        Returns
        -------
        np.ndarray
            Array of pairwise distances.
        """

        if reference is None:
            return self._compute_ensemble(ensemble)
        else:
            assert (
                self._format != "matrix"
            ), "Reference-based score only supports scalar and flat format"

            bi_ensembles = utils.get_bi_ensembles(ensemble, reference)
            scores = []
            for bi_ensemble in bi_ensembles:
                scores.append(self._compute_ensemble(bi_ensemble))

            match self._format:
                case "scalar":
                    return np.mean(scores)
                case "flat":
                    return np.array(scores)
                case _:
                    raise ValueError(f"Unknown strategy '{self._format}'")

    def _compute_ensemble(self, ensemble: Ensemble):
        scores = []

        index_pairs = list(itertools.combinations(range(len(ensemble.alignments)), r=2))
        for idx_x, idx_y in index_pairs:
            score = self.compute_similarity(
                ensemble.alignments[idx_x],
                ensemble.alignments[idx_y],
            )
            scores.append(score)

        match self._format:
            case "scalar":
                return np.mean(scores)
            case "flat":
                return np.array(scores)
            case "matrix":
                return utils.format_dist_mat(scores, ensemble)
            case _:
                raise ValueError(f"Unknown strategy '{self._format}'")

    def compute_similarity(
        self,
        alignment_x: Alignment,
        alignment_y: Alignment,
    ) -> float:
        """
        Compute the pairwise distance between two alignments.

        Parameters
        ----------
        alignment_x : Alignment
            First alignment for the computation.
        alignment_y : Alignment
            Second alignment for the computation.

        Returns
        -------
        float
            The computed distance.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.
        """
        raise NotImplementedError()

    def _init_key_map(
        self,
        alignment_x: Alignment,
        alignment_y: Alignment,
        key_x=None,
        key_y=None,
    ):
        """
        Initialize internal mapping from keys to alignments for caching.

        Parameters
        ----------
        alignment_x : Alignment
            First alignment.
        alignment_y : Alignment
            Second alignment.
        key_x : optional
            Key for the first alignment.
        key_y : optional
            Key for the second alignment.
        """
        self._key_x = alignment_x.key if key_x is None else key_x
        self._key_y = alignment_y.key if key_y is None else key_y
        self._alignment_map = {self._key_x: alignment_x, self._key_y: alignment_y}

    def _get_metric_prerequisites(self, key):
        """
        Prepare and return common prerequisites for metric computation.

        Parameters
        ----------
        key : hashable
            Key identifying the alignment.

        Returns
        -------
        tuple
            Tuple containing sequence array, index mapping, number of sequences,
            list of sequence lengths, and maximum sequence length.
        """
        S = np.array(self._alignment_map[key].msa)
        Q = encoding.gapped_index_mapping(S)
        K = len(S)
        L_k_list = [len(arr) for arr in Q]
        L_max = max(L_k_list)
        return S, Q, K, L_k_list, L_max

    def _get_from_cache(self, key, compute_cachables: callable, **kwargs):
        """
        Retrieve a value from cache or compute and store it if not present.

        Parameters
        ----------
        key : hashable
            Key for the cached value.
        compute_cachables : callable
            Function to compute the value if not cached.
        **kwargs
            Additional arguments for the compute function.

        Returns
        -------
        Any
            Cached or newly computed value.
        """
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
        """
        Check if all required values for the current metric are cached.

        Returns
        -------
        bool
            True if all required values are cached, False otherwise.
        """
        return (
            (self._cache is not None)
            and (self.name in self._cache)
            and (self._key_x in self._cache[self.name])
            and (self._key_y in self._cache[self.name])
        )


class _HomologySetMetric(_Metric):
    """
    Base class for metrics that operate on sets of homologous columns in alignments.
    """

    def _avg_jaccard_coef(self, encoding_enum: PositionalEncodingEnum):
        """
        Compute the average Jaccard coefficient-based distance between two alignments (used for SSP).

        Parameters
        ----------
        encoding_enum : PositionalEncodingEnum
            Encoding scheme for positions.

        Returns
        -------
        float
            Jaccard-based distance between the encoded alignments.
        """

        def from_scratch(key):
            S, Q, K, L_k_list, L_max = self._get_metric_prerequisites(key)
            A_code = encoding.encode_positions(S, encoding_enum)
            hcols_list, gap_mask_list = [], []
            for k in range(K):
                hcols_list.append(np.delete(A_code[:, Q[k]], k, axis=0))
                gap_mask_list.append(hcols_list[-1] != GAP_CODE)
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
        """
        Compute the average normalized Hamming distance between two alignments (used for D_seq, D_pos).

        Parameters
        ----------
        encoding_enum : PositionalEncodingEnum
            Encoding scheme for positions.

        Returns
        -------
        float
            Normalized Hamming distance between the encoded alignments.
        """

        def from_scratch(key):
            S, Q, K, L_k_list, L_max = self._get_metric_prerequisites(key)
            A_code = encoding.encode_positions(S, encoding_enum)
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
    """
    Computes the SSP (symmetrized sum-of-pairs) distance between two alignments.
    """

    enum = ME.D_SSP
    name = str(enum)

    def compute_similarity(self, alignment_x: Alignment, alignment_y: Alignment):
        """
        Compute the SSP distance between two alignments.

        Parameters
        ----------
        alignment_x : Alignment
            First alignment.
        alignment_y : Alignment
            Second alignment.

        Returns
        -------
        float
            The SSP distance.
        """
        self._init_key_map(alignment_x, alignment_y)
        return self._avg_jaccard_coef(PositionalEncodingEnum.UNIFORM)


class DSeqDistance(_HomologySetMetric):
    """
    Computes the "D_seq" distance between two alignments.
    """

    enum = ME.D_SEQ
    name = str(enum)

    def compute_similarity(self, alignment_x: Alignment, alignment_y: Alignment):
        """
        Compute the "D_seq" distance between two alignments.

        Parameters
        ----------
        alignment_x : Alignment
            First alignment.
        alignment_y : Alignment
            Second alignment.

        Returns
        -------
        float
            The "D_seq" distance.
        """
        self._init_key_map(alignment_x, alignment_y)
        return self._avg_hamming_dist(PositionalEncodingEnum.SEQUENCE)


class DPosDistance(_HomologySetMetric):
    """
    Computes the "D_pos" distance between two alignments.
    """

    enum = ME.D_POS
    name = str(enum)

    def compute_similarity(self, alignment_x: Alignment, alignment_y: Alignment):
        """
        Compute the "D_pos" distance between two alignments.

        Parameters
        ----------
        alignment_x : Alignment
            First alignment.
        alignment_y : Alignment
            Second alignment.

        Returns
        -------
        float
            The D_pos distance.
        """
        self._init_key_map(alignment_x, alignment_y)
        return self._avg_hamming_dist(PositionalEncodingEnum.POSITION)


class PHashDistance(_Metric):
    """
    Computes the perceptual hash-based Hamming distance between two alignments.

    Attributes
    ----------
    enum : MethodEnum
        Enum identifier for the metric.
    name : str
        Name of the metric, includes hash size.
    _hash_size : int
        Size of the hash in bits.
    """

    enum = ME.D_PHASH
    name = None

    def __init__(
        self,
        hash_size: int = 16,
        format: Literal["scalar", "flat", "matrix"] = "scalar",
        cache: dict = None,
    ):
        """
        Initialize the perceptual hash distance metric.

        Parameters
        ----------
        hash_size : int, default=16
            Size of the perceptual hash in bits.
        format : {"scalar", "flat", "matrix"}, default="scalar"
            Output format: scalar, flat list, or square matrix.
        cache : dict, optional
            Dictionary for caching intermediate results.
            Caching is applied by default. The parameter allows for re-using an existing cache.
        """
        self.name = self.enum + f"_{hash_size}bit"
        self._hash_size = hash_size
        self._format = format
        self._cache = {} if cache is None else cache

    def compute_similarity(self, alignment_x: Alignment, alignment_y: Alignment):
        """
        Compute the perceptual hash-based Hamming distance between two alignments.

        Parameters
        ----------
        alignment_x : Alignment
            First alignment.
        alignment_y : Alignment
            Second alignment.

        Returns
        -------
        float
            Normalized Hamming distance between the hashes.
        """
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
        """
        Pad an MSA array to the specified number of columns with gap characters.

        Parameters
        ----------
        msa : np.ndarray
            Multiple sequence alignment array.
        length : int
            Desired number of columns.

        Returns
        -------
        np.ndarray
            Padded MSA array.
        """
        cols_to_pad = length - msa.shape[1]
        if not cols_to_pad:
            return msa
        return np.concatenate(
            [msa, np.full((len(msa), cols_to_pad), fill_value=GAP_CHAR)], axis=1
        )
