import numpy as np
from typing import Literal
from tqdm import tqdm
from abc import ABC
from aldiscore.enums.enums import PositionalEncodingEnum
from aldiscore.datastructures.ensemble import Ensemble
from aldiscore.scoring import encoding


class _ConfusionScore(ABC):
    def __init__(
        self,
        aggregate: Literal["site", "sequence", None] = None,
        code_dtype: np.dtype = np.int32,
        verbose: bool = False,
    ):
        super().__init__()
        self._aggregate = aggregate
        self._dtype = code_dtype
        self._verbose = verbose

    def compute(self, ensemble: Ensemble) -> float | np.ndarray | list[np.ndarray]:
        self._compute_prerequisites(ensemble)
        self._compute_bespoke_arguments()
        return self._confusion()

    def _seq_confusion(self):
        raise NotImplementedError()

    def _compute_bespoke_arguments(self):
        pass

    def _compute_prerequisites(self, ensemble: Ensemble):
        self._K = len(ensemble.dataset.records)
        self._I = len(ensemble.alignments)
        self._N_k_list = ensemble.dataset._sequence_lengths.copy()
        self._L_max = max(self._N_k_list)
        self._N_max = max([alignment.shape[1] for alignment in ensemble.alignments])
        self._A_code_list = []
        self._Q_list = []
        self._dtype = self._dtype

        for i in range(self._I):
            # Prepare encodings and mappings from unaligned to aligned index
            A = np.array(ensemble.alignments[i].msa)
            self._Q_list.append(encoding.gapped_index_mapping(A, self._dtype))
            A_code = encoding.encode_positions(
                A, PositionalEncodingEnum.POSITION, self._dtype
            )
            self._A_code_list.append(A_code)

    def _compute_replication_sets(self, k: int):
        rcols = np.empty((self._N_k_list[k], self._K, self._I), dtype=self._dtype)
        for i in range(self._I):
            rcols[:, :, i] = self._A_code_list[i][:, self._Q_list[i][k]].T
        return rcols

    def _aggregate_site_vals(self, site_vals, aggregate):
        if aggregate == "site":
            out = np.mean(np.concatenate(site_vals))
        elif aggregate == "sequence":
            out = np.array([np.mean(seq_dists) for seq_dists in site_vals])
        elif aggregate is None:
            return [seq_dists for seq_dists in site_vals]
        else:
            raise ValueError(f"Unknown strategy {aggregate}")
        return out

    def _confusion(self):
        site_vals = []
        K_range = tqdm(list(range(self._K))) if self._verbose else range(self._K)
        for k in K_range:
            # For each k, rcols has dimensions (N_k, K, I)
            rcols = self._compute_replication_sets(k)
            seq_confusion = self._seq_confusion(rcols, k)
            site_vals.append(seq_confusion)

        return self._aggregate_site_vals(site_vals, self._aggregate)


class ConfusionSet(_ConfusionScore):
    def _seq_confusion(self, rcols, k):
        # These comments describe what is going on below. The shape of the current array is on the left
        # (N_k, K, I)   -> np.delete (delete kth entry from homology column to remove the site itself) ->
        # (N_k, K-1, I) -> np.sort   (sort homology column values, to have incrementing values for diff) ->
        # (N_k, K-1, I) -> np.diff   (using diff to compute number of unique entries encoded as a boolean array) ->
        # (N_k, K-1, I) -> np.sum    (sum over diff output to get sum of unique values in each replicate set) ->
        # (N_k, K-1)    -> np.mean   (average the replicate sets of each site of the homology column) ->
        # (N_k)
        seq_confusion = (
            np.mean(
                np.sum(
                    np.diff(
                        np.sort(
                            np.delete(rcols, k, axis=1),
                            axis=2,
                        ),
                        n=1,
                        prepend=0,
                        axis=2,
                    )
                    != 0,
                    axis=2,
                ),
                axis=1,
            ),
            # Formatted as a tuple for readability, extracting singleton entry here
        )[0]

        seq_confusion = (seq_confusion - 1) / (self._I - 1)

        return seq_confusion


class ConfusionEntropy(_ConfusionScore):
    def _compute_bespoke_arguments(self):
        uniform = np.full(shape=self._I, fill_value=(1 / self._I))
        self._max_entropy = -np.sum(uniform * np.log2(uniform))

    def _seq_confusion(self, rcols, k) -> np.ndarray:
        # TODO: Add comments to make this comprehensible
        # Compute entropy over replicate set ("I" dimension), average over replicate sets of homology column
        N_k = self._N_k_list[k]
        rep_cols = np.sort(np.delete(rcols, k, axis=1), axis=2)
        append_vals = rep_cols[:, :, -1, np.newaxis] + 1
        end_mask = np.diff(rep_cols, n=1, append=append_vals, axis=2) != 0
        site_counts = np.tile(np.arange(1, self._I + 1), (N_k, self._K - 1, 1))
        site_counts[~end_mask] = 0
        site_counts = encoding._ffill_numpy_3d_axis_2(site_counts, val=0)
        pred_counts = np.roll(site_counts, shift=1, axis=2)
        pred_counts[:, :, 0] = 0
        counts = site_counts - pred_counts

        percentages = counts / np.sum(counts, axis=2, keepdims=True)
        percentages[~end_mask] = 1  # Avoiding -inf values in log2

        seq_confusion = -percentages * np.log2(percentages)
        seq_confusion[~end_mask] = 0
        seq_confusion = seq_confusion.sum(axis=2).mean(axis=1)

        # Normalize with maximal entropy
        seq_confusion = seq_confusion / self._max_entropy

        return seq_confusion


class ConfusionDisplace(_ConfusionScore):
    def __init__(
        self,
        thresholds: list[int] = [0, 1, 2, 4, 8, 16, 32],
        weights: list[float] = [1, 1, 1, 1, 1, 1, 1],
        aggregate: Literal["site", "sequence", None] = None,
        code_dtype: np.dtype = np.int32,
        verbose: bool = False,
    ):
        super().__init__(aggregate, code_dtype, verbose)
        self._thresholds = np.array(thresholds)
        self._norm_weights = np.array(weights) / np.sum(weights)

    def _seq_confusion(self, rcols, k):
        temp = np.delete(rcols, k, axis=1).clip(min=0)
        res_mask = (temp > 0).copy()
        n_residues = res_mask.sum(axis=2)
        avg_mask = res_mask.copy()
        avg_mask[n_residues == 0] = 1
        temp = temp - (
            np.sum(temp, axis=2, keepdims=True)
            / np.maximum(n_residues, 1)[:, :, np.newaxis]
        )
        # Compute standard deviation
        seq_confusion = np.sqrt(
            (temp**2).sum(where=avg_mask, axis=2) / np.maximum(n_residues - 1, 1)
        )
        seq_confusion = (seq_confusion[:, :, np.newaxis] > self._thresholds).dot(
            self._norm_weights
        )
        seq_confusion[n_residues <= 1] = 0
        seq_confusion = seq_confusion.mean(axis=1)

        return seq_confusion
