import numpy as np
from typing import Literal, Optional
import itertools
import math
from aldiscore.scoring.encoding import _ffill_numpy_2d_axis_1
from collections import defaultdict
from functools import partial
import itertools
from pathlib import Path
import os
import pandas as pd
from aldiscore import RSTATE


def sample_index_tuples(n: int, r: int, k: int, seed: int):
    """
    This is necessary because itertools.combinations is inefficient for sampling + combinatorics.
    - n: index range
    - r: tuple length
    - k: (maximum) number of tuples
    """
    samples = set()
    max_comb = math.comb(n, r)
    limit = np.minimum(max_comb, k)
    # prioritize_new = n < limit # TODO: implement or remove
    while True:
        np.random.seed(seed)
        # Ensure that seed is incremented, otherwise infinite loop here
        seed += 1

        # Sample k r-tuples randomly -> (k,r)
        samples_new = np.sort(
            np.random.randint(low=0, high=n, size=2 * k * r).reshape(2 * k, r), axis=1
        )
        # Remove rows where two or more positions are identical
        diff_mask = (np.diff(samples_new, axis=1) != 0).all(axis=1)
        samples_new = samples_new[diff_mask].tolist()
        # print(samples_new)
        samples_new = list(map(lambda tup: tuple(sorted(tup)), samples_new))
        samples.update(samples_new)

        if len(samples) >= limit:
            break

    return list(samples)[:k]


def compute_gap_lengths(alignment: np.ndarray, gap_code) -> np.ndarray:
    """Computes the gap lengths in a given alignment. Non-gapped positions are encoded with gap_code."""
    gap_mask = alignment == gap_code
    gap_ends = np.diff(gap_mask, n=1, axis=1, append=0) == -1
    # gap_counts contains an incrementing counter for gapped positions in each sequence
    gap_counts = gap_mask.cumsum(axis=1)
    # Set all positions to zero that are not the end of a gap region
    gap_counts[~gap_ends] = 0
    # For each gap region, the last gap now holds the count of all gaps up to that point
    # Right-shift the gap end counters (counts[i] = counts_shifted[i+1])
    gap_end_counts_shifted = np.roll(gap_counts, shift=1, axis=1)
    # Set the first position to zero (no preceding gap regions)
    gap_end_counts_shifted[:, 0] = 0
    # Forward fill the values: now position i holds the cumulative gap count prior to the current gap region
    gap_end_counts_shifted = _ffill_numpy_2d_axis_1(gap_end_counts_shifted, val=0)
    # Compute the length of the gap regions at their last position
    site_codes = gap_counts - gap_end_counts_shifted
    # Remove all positions that are not gap region ends
    site_codes[~gap_ends] = 0
    return site_codes


def repeat_distributions(seq_arrs: list[np.ndarray]) -> tuple[np.ndarray]:
    """
    Occurrences of homopolymers and their lengths.
    """
    count_table = defaultdict(partial(np.zeros, shape=len(seq_arrs)))
    len_table = defaultdict(partial(np.zeros, shape=len(seq_arrs)))
    for i, seq in enumerate(seq_arrs):

        # Homopolymers via groupby
        for _, group in itertools.groupby(seq):
            repeat = tuple(group)
            k = len(repeat)
            # if k > 1:
            count_table[hash(repeat)][i] += 1
            len_table[k][i] += 1

    count_arr = np.stack(list(count_table.values()), axis=1)
    len_arr = np.stack(list(len_table.values()), axis=1)
    count_arr = count_arr / count_arr.sum(axis=1, keepdims=True)
    len_arr = len_arr / len_arr.sum(axis=1, keepdims=True)
    return count_arr, len_arr


def shannon_entropy(probs: np.ndarray, axis: int = None):
    dist = probs / probs.sum(axis=axis, keepdims=axis is not None)
    return -np.sum(dist * np.log2(dist), axis=axis)


def js_divergence(p: np.ndarray, q: np.ndarray, axis: int):
    p = p / np.sum(p, axis=axis, keepdims=True)
    q = q / np.sum(q, axis=axis, keepdims=True)
    m = (p + q) / 2
    # Relative entropy must always be non-negative --> enforce with clip
    left = np.sum(p * np.log(p / m), axis=1).clip(min=0)
    right = np.sum(q * np.log(q / m), axis=1).clip(min=0)
    js = np.sqrt((left + right) / 2)
    return js


def load_features(
    data_dir: Path,
    exclude_sources: list = None,
    include_sources: list = None,
    label_scale: Optional[float] = "auto",
    exclude_features: list = None,
    include_features: list = None,
    drop_na: bool = True,
) -> tuple[pd.DataFrame]:
    assert_msg = "Specify either 'exclude_{0}' or 'include_{0}'"
    assert not (exclude_sources and include_sources), assert_msg.format("sources")
    assert not (exclude_features and include_features), assert_msg.format("features")

    sources = os.listdir(data_dir)
    if exclude_sources:
        sources = list(filter(lambda s: s not in exclude_sources, sources))
    elif include_sources:
        sources = list(filter(lambda s: s in include_sources, sources))
    feat_dfs = []
    label_dfs = []
    for source in sources:
        feat_df = pd.read_parquet(data_dir / source / "features.parquet")
        label_df = pd.read_parquet(data_dir / source / "stats.parquet")
        label_df = label_df[["mean"]].query("method == 'dpos'").droplevel(2)
        feat_dfs.append(feat_df)
        label_dfs.append(label_df)

    feat_df = pd.concat(feat_dfs, axis=0).sort_index()
    label_df = pd.concat(label_dfs, axis=0).sort_index()

    if label_scale == "auto":
        # Scales up labels by only ~5% IF there are very difficult datasets
        label_df = label_df / label_df.max()
    else:
        label_df = (label_df * label_scale).clip(upper=1)

    cols = feat_df.columns
    if exclude_features:
        mask = np.full(len(cols), True)
        for col in exclude_features:
            mask[cols.str.contains(col)] = False
        cols = cols[mask]
    elif include_features:
        mask = np.full(len(cols), False)
        for col in include_features:
            mask[cols.str.contains(col)] = True
        cols = cols[mask]

    drop_df = feat_df.drop(cols, axis=1).copy()
    feat_df = feat_df[cols]

    # Left or Right Join if rows have been removed
    if len(label_df) > len(feat_df):
        label_df = label_df.loc[feat_df.index]
    else:
        feat_df = feat_df.loc[label_df.index]
        drop_df = drop_df.loc[label_df.index]

    assert (feat_df.index == label_df.index).all()

    if drop_na:
        nan_mask = feat_df.isna().any(axis=1) | label_df.isna().any(axis=1)
        print(f"Dropping {sum(nan_mask)} NaN rows...")
        feat_df = feat_df[~nan_mask]
        label_df = label_df[~nan_mask]

    return feat_df, drop_df, label_df


def train_test_valid_split(index: pd.Index):
    from sklearn.model_selection import train_test_split

    train_idxs, test_idxs = train_test_split(
        index.to_list(), test_size=0.2, random_state=RSTATE
    )
    test_idxs, valid_idxs = train_test_split(
        test_idxs, test_size=0.5, random_state=RSTATE
    )
    return train_idxs, valid_idxs, test_idxs


def compute_metrics(model, X, y, eps):
    y_pred = model.predict(X)
    perf_dicts = []
    rmse = (((y_pred - y) ** 2).sum() / len(y)) ** 0.5
    rmse_cv = rmse / np.mean(y)
    mae = (np.abs(y_pred - y)).sum() / len(y)
    mape = (np.abs(y_pred - y) / (y + eps)).sum() / len(y)
    mape_p50 = np.percentile(np.abs(y_pred - y) / (y + eps), 50)
    corr = np.corrcoef(y, y_pred)[0, 1]
    perf_dict = {}
    perf_dict[f"RMSE"] = f"{rmse:.4f}"
    perf_dict[f"RMSE_CV"] = f"{rmse_cv:.4f}"
    perf_dict[f"MAE"] = f"{mae:.4f}"
    perf_dict[f"MAPE"] = f"{mape:.4f}"
    perf_dict[f"MAPE_P50"] = f"{mape_p50:.4f}"
    perf_dict[f"CORR"] = f"{corr:.4f}"
    perf_dicts.append(perf_dict)

    perf_df = pd.DataFrame(perf_dicts)
    return perf_df
