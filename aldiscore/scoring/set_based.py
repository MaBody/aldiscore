"""
Implements set-based "confusion scores" for quantifying agreement in an ensemble of alignments.

Provides three main variants:
- ConfusionSet: Agreement based on unique sets of aligned residues.
- ConfusionEntropy: Agreement based on entropy across replicates.
- ConfusionDisplace: Agreement based on displacement (spread) of aligned residues.

These scores operate by aggregating information across all alignments in the ensemble,
using different strategies to summarize the consistency of residue placement.
"""

import numpy as np
from typing import Literal
from tqdm import tqdm
from abc import ABC
from aldiscore.enums.enums import PositionalEncodingEnum
from aldiscore.datastructures.ensemble import Ensemble
from aldiscore.datastructures.alignment import Alignment
from aldiscore.scoring import encoding
from aldiscore.scoring import utils


class _ConfusionScore(ABC):
    """
    Base class for set-based confusion scores on alignment ensembles.

    Handles aggregation and preparation of encoded alignment data for downstream
    agreement/confusion calculations.

    Parameters
    ----------
    format : {"scalar", "sequence", "residue"}, optional
        Aggregation strategy for the output (default: "scalar").
    verbose : bool, optional
        Whether to show progress bars (default: False).
    """

    def __init__(
        self,
        format: Literal["scalar", "sequence", "residue"] = "scalar",
        verbose: bool = False,
    ):
        super().__init__()
        self._format = format
        self._verbose = verbose

    def compute(
        self, ensemble: Ensemble, reference: Alignment = None
    ) -> float | np.ndarray | list[np.ndarray]:
        """
        Compute the confusion score for an ensemble of alignments.

        Parameters
        ----------
        ensemble : Ensemble
            Ensemble of alignments to evaluate.
        reference : Alignment, optional
            If provided, computes average confusion with respect to reference.

        Returns
        -------
        float or np.ndarray or list[np.ndarray]
            The confusion score(s), aggregated as specified.
        """
        if reference is None:
            self._compute_prerequisites(ensemble)
            self._compute_bespoke_arguments()
            return self._confusion()
        else:
            assert (
                self._format == "scalar"
            ), "Reference-based score only supports scalar format"
            bi_ensembles = utils.get_bi_ensembles(ensemble, reference)
            scores = []
            for bi_ensemble in bi_ensembles:
                self._compute_prerequisites(bi_ensemble)
                self._compute_bespoke_arguments()
                scores.append(self._confusion())
            return np.mean(scores)

    def _seq_confusion(self):
        """
        Compute the confusion score for a single sequence (to be implemented by subclasses).
        """
        raise NotImplementedError()

    def _compute_bespoke_arguments(self):
        """
        Compute any subclass-specific arguments needed for the confusion calculation.
        """
        pass

    def _compute_prerequisites(self, ensemble: Ensemble):
        """
        Prepare encoded alignments and index mappings for all alignments in the ensemble.

        Parameters
        ----------
        ensemble : Ensemble
            Ensemble of alignments to process.
        """
        self._K = len(ensemble.dataset.records)
        self._I = len(ensemble.alignments)
        self._N_k_list = ensemble.dataset._sequence_lengths.copy()
        self._L_max = max(self._N_k_list)
        self._N_max = max([alignment.shape[1] for alignment in ensemble.alignments])
        self._A_code_list = []
        self._Q_list = []

        for i in range(self._I):
            # Prepare encodings and mappings from unaligned to aligned index
            A = np.array(ensemble.alignments[i].msa)
            self._Q_list.append(encoding.gapped_index_mapping(A))
            A_code = encoding.encode_positions(A, PositionalEncodingEnum.POSITION)
            self._A_code_list.append(A_code)

    def _compute_replication_sets(self, k: int):
        """
        For a given sequence index k, collect the replication sets for each residue in that sequence.
        The confusion of each residue is based on the agreement on its aligned positions.
        Each residue is aligned with K-1 residues per replicate.
        Including the given index k (removed later), that results in a K x I matrix per residue (K replication sets).


        Parameters
        ----------
        k : int
            Sequence index.

        Returns
        -------
        np.ndarray
            Array of shape (N_k, K, I) containing replication sets for sequence k.
        """
        rcols = np.empty((self._N_k_list[k], self._K, self._I), dtype=np.int32)
        for i in range(self._I):
            rcols[:, :, i] = self._A_code_list[i][:, self._Q_list[i][k]].T
        return rcols

    def _aggregate_site_vals(self, site_vals, format):
        """
        Aggregate confusion values across sites or sequences.

        Parameters
        ----------
        site_vals : list
            List of confusion values per site.
        format : {"scalar", "sequence", "residue"}
            Aggregation strategy.

        Returns
        -------
        float or np.ndarray or list[np.ndarray]
            Aggregated confusion score(s).
        """
        if format == "scalar":
            out = np.concatenate(site_vals).mean(dtype=float)
        elif format == "sequence":
            out = [float(sum(seq_dists)) / len(seq_dists) for seq_dists in site_vals]
        elif format == "residue":
            return [seq_dists.tolist() for seq_dists in site_vals]
        else:
            raise ValueError(f"Unknown strategy {format}")
        return out

    def _confusion(self):
        """
        Compute the confusion score for all sequences in the ensemble.

        Returns
        -------
        float or np.ndarray or list
            Aggregated confusion score(s).
        """
        site_vals = []
        K_range = tqdm(list(range(self._K))) if self._verbose else range(self._K)
        for k in K_range:
            # For each k, rcols has dimensions (N_k, K, I)
            rcols = self._compute_replication_sets(k)
            seq_confusion = self._seq_confusion(rcols, k)
            site_vals.append(seq_confusion)

        return self._aggregate_site_vals(site_vals, self._format)


class ConfusionSet(_ConfusionScore):
    """
    Computes the set-based confusion score for an ensemble of alignments.

    Measures agreement by counting the average number of unique residues in the replication sets,
    normalized by the number of alignments. Lower values indicate higher agreement.
    """

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
    """
    Computes the entropy-based confusion score for an ensemble of alignments.

    Measures agreement by calculating the average entropy of aligned residue distributions per replication set,
    normalized by the maximal possible entropy for a set of I samples.
    Lower values indicate higher agreement.
    """

    def _compute_bespoke_arguments(self):
        uniform = np.full(shape=self._I, fill_value=(1 / self._I))
        self._max_entropy = -np.sum(uniform * np.log2(uniform))

    def _seq_confusion(self, rcols, k) -> np.ndarray:
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
    """
    Computes the displacement-based confusion score for an ensemble of alignments.

    Measures agreement by quantifying the spread (standard deviation) of aligned residue positions
    across replicates, using user-defined thresholds and weights. Lower values indicate higher agreement.

    Parameters
    ----------
    format : {"scalar", "sequence", "residue"}, optional
        Aggregation strategy for the output (default: "scalar").
    thresholds : list[int], optional
        List of thresholds for displacement (default: [0, 1, 2, 4, 8, 16, 32]).
    weights : list[float], optional
        Weights for each threshold (default: uniform).
    verbose : bool, optional
        Whether to show progress bars (default: False).
    """

    def __init__(
        self,
        format: Literal["scalar", "sequence", "residue"] = "scalar",
        thresholds: list[int] = [0, 1, 2, 4, 8, 16, 32],
        weights: list[float] = [1, 1, 1, 1, 1, 1, 1],
        verbose: bool = False,
    ):
        super().__init__(format, verbose)
        self._thresholds = np.array(thresholds, dtype=np.int32)
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
