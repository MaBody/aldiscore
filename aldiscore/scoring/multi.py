from enums.enums import PositionalEncodingEnum, FeatureEnum
from datastructures.ensemble import Ensemble
import scoring.encoding
import numpy as np
from typing import Literal
from tqdm import tqdm


# class ConfusionScore:
#     enum = None
#     name = None

#     def __init__(self):
#         pass

#     def compute(self, ensemble: Ensemble):
#         pass


# class ConfSet(ConfusionScore):
#     enum = FeatureEnum.CONFUSION_SET
#     name = str(enum)

#     def compute(self, ensemble:Ensemble):
#         def _init_standard_data()


def confusion_score(
    ensemble: Ensemble,
    encoding: PositionalEncodingEnum,
    method: FeatureEnum = FeatureEnum.CONFUSION_ENTROPY,
    aggregate: Literal["site", "sequence", None] = None,
    normalize: bool = True,
    dtype: np.dtype = np.int32,
    verbose: bool = False,
    thresholds=[0, 1, 2, 4, 8, 16, 32],
    weights=[1, 1, 1, 1, 1, 1, 1],
):
    """For each site, compute the number of different unique sites that replicates propose for each site of the homology set.
    Average over all sites in the homology set."""

    K = len(ensemble.dataset.sequences)
    I = len(ensemble.ensemble)
    N_k_list = ensemble.dataset._sequence_lengths.copy()
    L_max = max(N_k_list)
    N_max = max([msa.number_of_sites() for msa in ensemble.ensemble])
    A_code_list = []
    Q_list = []

    max_entropy = None
    if (method == FeatureEnum.CONFUSION_ENTROPY) and normalize:
        uniform = np.full(shape=I, fill_value=1 / I)
        max_entropy = -np.sum(uniform * np.log2(uniform))

    thresholds_ = None
    norm_weights = None
    if method == FeatureEnum.CONFUSION_DISPLACEMENT:
        thresholds_ = np.array(thresholds)
        norm_weights = np.array(weights) / np.sum(weights)

    for i in range(I):
        # Prepare encodings and mappings from unaligned to aligned index
        A = np.array(ensemble.ensemble[i].msa)
        Q_list.append(scoring.encoding.gapped_index_mapping(A, dtype))
        A_code = scoring.encoding.encode_positions(A, L_max, encoding, dtype)
        A_code_list.append(A_code)

    site_vals = []
    K_range = tqdm(list(range(K))) if verbose else range(K)
    for k in K_range:
        # For each k, hcols has dimensions (N_k, K, I)
        rcols = np.empty((N_k_list[k], K, I), dtype=dtype)
        for i in range(I):
            rcols[:, :, i] = A_code_list[i][:, Q_list[i][k]].T

        if method == FeatureEnum.CONFUSION_SET:
            # Collect the number of distinct homologous positions per site across all replicates
            seq_confusion = _seq_confusion_set(rcols, k, I, normalize)

        elif method == FeatureEnum.CONFUSION_ENTROPY:
            # Compute the entropy per site across all replicates
            seq_confusion = _seq_confusion_entropy(
                rcols, K, k, N_k_list[k], I, max_entropy, normalize
            )
        elif method == FeatureEnum.CONFUSION_DISPLACEMENT:
            seq_confusion = _seq_confusion_displacement(
                rcols, k, thresholds_, norm_weights
            )
        # elif method == "composite":
        #     # Compute the entropy per site across all replicates
        #     seq_confusion_s = _seq_confusion_set(rcols, k, I, normalize)
        #     seq_confusion_d = _seq_confusion_displacement(
        #         rcols, k, N_max, thresholds_, norm_weights
        #     )
        #     seq_confusion = seq_confusion_s * seq_confusion_d
        else:
            raise ValueError(f"Unkown scoring method '{method}'")

        site_vals.append(seq_confusion)
    if aggregate == "site":
        dist = np.mean(np.concatenate(site_vals))
    elif aggregate == "sequence":
        dist = np.array([np.mean(seq_dists) for seq_dists in site_vals])
    elif aggregate is None:
        return [seq_dists for seq_dists in site_vals]
    else:
        raise ValueError(f"Unknown strategy {aggregate}")
    return dist


def _seq_confusion_set(rcols, k, I, normalize):
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

    if normalize:
        seq_confusion = (seq_confusion - 1) / (I - 1)

    return seq_confusion


def _seq_confusion_entropy(rcols, K, k, N_k, I, max_entropy, normalize):
    # TODO: Add comments to make this comprehensible
    # Compute entropy over replicate set ("I" dimension), average over replicate sets of homology column
    rep_cols = np.sort(np.delete(rcols, k, axis=1), axis=2)
    append_vals = rep_cols[:, :, -1, np.newaxis] + 1
    end_mask = np.diff(rep_cols, n=1, append=append_vals, axis=2) != 0

    site_counts = np.tile(np.arange(1, I + 1), (N_k, K - 1, 1))
    site_counts[~end_mask] = 0
    site_counts = scoring.encoding._ffill_numpy_3d_axis_2(site_counts, val=0)
    pred_counts = np.roll(site_counts, shift=1, axis=2)
    pred_counts[:, :, 0] = 0
    counts = site_counts - pred_counts

    percentages = counts / np.sum(counts, axis=2, keepdims=True)
    percentages[~end_mask] = 1  # Avoiding -inf values in log2

    seq_confusion = -percentages * np.log2(percentages)
    seq_confusion[~end_mask] = 0
    seq_confusion = seq_confusion.sum(axis=2).mean(axis=1)

    if normalize:
        seq_confusion = seq_confusion / max_entropy

    return seq_confusion


def _seq_confusion_displacement(rcols, k, thresholds, weights):
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
    seq_confusion = (seq_confusion[:, :, np.newaxis] > thresholds).dot(weights)
    seq_confusion[n_residues <= 1] = 0
    seq_confusion = seq_confusion.mean(axis=1)

    return seq_confusion
