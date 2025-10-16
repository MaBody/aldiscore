import numpy as np
from typing import List, Tuple, Optional
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
    Sample k r-tuples from range(n) without replacement.

    More efficient than using itertools.combinations for large n and r.

    Args:
        n: Size of index range to sample from
        r: Length of each tuple
        k: Number of tuples to sample
        seed: Random seed for reproducibility

    Returns:
        List of k sorted tuples, each containing r unique indices

    Note:
        Uses rejection sampling - generates more tuples than needed
        and filters out those with duplicate indices.
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
    """
    Compute the length of gap regions in sequence alignments.

    For each position that ends a gap region, stores the length of that gap.
    All other positions contain 0.

    Args:
        alignment: 2D array of sequences, gaps marked with gap_code
        gap_code: Integer code used to represent gaps

    Returns:
        2D array same shape as alignment, where each position ending
        a gap region contains the length of that gap, others contain 0
    """
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


def repeat_distributions(seq_arrs: List[np.ndarray]) -> Tuple[np.ndarray]:
    """
    Compute distributions of repeated elements (homopolymers) in sequences.

    For each sequence, finds:
    1. Distribution of specific repeat patterns (e.g., AAA vs TTT)
    2. Distribution of repeat lengths (e.g., how many 2-mers vs 3-mers)

    Args:
        seq_arrs: List of sequences as integer arrays

    Returns:
        Tuple of two arrays:
        - count_arr: Normalized counts of each specific repeat pattern
        - len_arr: Normalized counts of each repeat length
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
    count_arr = np.divide(
        count_arr, count_arr.sum(axis=1, keepdims=True), dtype=np.float32
    )
    len_arr = np.divide(len_arr, len_arr.sum(axis=1, keepdims=True), dtype=np.float32)
    return count_arr, len_arr


def shannon_entropy(probs: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Compute Shannon entropy of probability distributions.

    Args:
        probs: Array of probability values
        axis: Axis to sum over when computing entropy
              If None, compute total entropy

    Returns:
        Array of entropy values. If axis is None, returns a scalar.
    """
    dist = probs / probs.sum(axis=axis, keepdims=axis is not None)
    return -np.sum(dist * np.log2(dist), axis=axis)


# def js_divergence(p: np.ndarray, q: np.ndarray, axis: int):
#     p = p / np.sum(p, axis=axis, keepdims=True)
#     q = q / np.sum(q, axis=axis, keepdims=True)
#     m = (p + q) / 2
#     # Relative entropy must always be non-negative --> enforce with clip
#     left = np.sum(p * np.log(p / m), axis=1).clip(min=0)
#     right = np.sum(q * np.log(q / m), axis=1).clip(min=0)
#     js = np.sqrt((left + right) / 2)
#     return js


def js_divergence(dist: np.ndarray, axis=1, chunk_size=5e7):
    """
    Compute pairwise Jensen-Shannon divergence between probability distributions.

    Efficiently computes JS divergence between all pairs of distributions in dist.
    Uses chunking to handle large distributions memory-efficiently.

    Args:
        dist: Array of probability distributions, shape (n_distributions, n_bins)
        axis: Axis containing the probability values (default=1)
        chunk_size: Maximum number of values to process at once
                   Lower values use less memory but compute slower

    Returns:
        Flat array of JS divergences between all pairs.
    """
    idxs = np.array(list(itertools.combinations(range(len(dist)), r=2)))
    n_pairs = idxs.shape[0]
    n_bins = dist.shape[1]
    results = []
    step_size = max(1, int(chunk_size // n_bins))
    # Process in chunks
    for start in range(0, n_pairs, step_size):
        end = min(start + step_size, n_pairs)
        p = dist[idxs[start:end, 0]]
        q = dist[idxs[start:end, 1]]
        p = p / np.sum(p, axis=axis, keepdims=True)
        q = q / np.sum(q, axis=axis, keepdims=True)
        m = (p + q) / 2
        # Relative entropy must always be non-negative --> enforce with clip
        left = np.sum(p * np.log(p / m), axis=1).clip(min=0)
        right = np.sum(q * np.log(q / m), axis=1).clip(min=0)
        js = np.sqrt((left + right) / 2)
        results.append(js)
    return np.concatenate(results)


def load_features(
    data_dir: Path,
    exclude_sources: list = None,
    include_sources: list = None,
    label_scale: Optional[float] = "auto",
    exclude_features: list = None,
    include_features: list = None,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load feature and label data for model training.

    Args:
        data_dir: Directory containing feature data
        exclude_sources: List of source files to exclude
        include_sources: List of source files to include (mutually exclusive with exclude)
        label_scale: How to scale labels:
                    - "auto": Scale by maximum value
                    - float: Scale by this value
        exclude_features: List of feature patterns to exclude
        include_features: List of feature patterns to include
        drop_na: Whether to drop rows with missing values

    Returns:
        Tuple of DataFrames:
        - Features used for training
        - Dropped features
        - Labels (difficulty scores)

    Raises:
        AssertionError: If both include and exclude are specified
    """
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

    # Convert singleton dataframe to series
    labels = label_df.iloc[:, 0]

    return feat_df, drop_df, labels


def train_test_valid_split(index: pd.Index):
    """
    Split data indices into train, validation and test sets.

    Uses a 80/10/10 split ratio:
    1. Split into 80% train, 10% test+valid
    2. Split test+valid into 50% test, 50% valid

    Args:
        index: Pandas index to split

    Returns:
        Three lists of indices for train, validation and test sets
    """
    from sklearn.model_selection import train_test_split

    train_idxs, test_idxs = train_test_split(
        index.to_list(), test_size=0.2, random_state=RSTATE
    )
    test_idxs, valid_idxs = train_test_split(
        test_idxs, test_size=0.5, random_state=RSTATE
    )
    return train_idxs, valid_idxs, test_idxs


def compute_metrics(model, X, y):
    """
    Compute regression metrics for model evaluation.

    Calculates:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² score
    - Correlation coefficient

    Args:
        model: Trained model with predict() method
        X: Feature matrix
        y: True target values

    Returns:
        DataFrame with computed metrics
    """
    from sklearn.metrics import r2_score

    y_pred = model.predict(X)
    perf_dicts = []
    rmse = (((y_pred - y) ** 2).sum() / len(y)) ** 0.5
    mae = (np.abs(y_pred - y)).sum() / len(y)
    corr = np.corrcoef(y, y_pred)[0, 1]
    perf_dict = {}
    perf_dict["RMSE"] = f"{rmse:.4f}"
    perf_dict["MAE"] = f"{mae:.4f}"
    perf_dict["R^2"] = f"{r2_score(y, y_pred):.4f}"
    perf_dict["CORR"] = f"{corr:.4f}"
    perf_dicts.append(perf_dict)

    perf_df = pd.DataFrame(perf_dicts)
    return perf_df


def optuna_search(
    X,
    y,
    n_trials: int = 100,
    n_estimators: int = 500,
    n_jobs: int = 1,
):
    """
    Perform hyperparameter optimization for LightGBM model using Optuna.

    Uses k-fold cross validation with early stopping to find optimal parameters.
    Optimizes for RMSE on validation set.

    Args:
        X: Feature matrix
        y: Target values
        n_trials: Number of parameter combinations to try
        n_estimators: Maximum number of boosting rounds
        n_jobs: Number of parallel jobs for optimization

    Returns:
        Optimized LGBMRegressor model

    Note:
        Tunes the following parameters:
        - subsample
        - learning_rate
        - colsample_bytree
        - feature_fraction_bynode
        - min_child_samples
        - num_leaves
        - reg_alpha
        - reg_lambda
    """
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    _N_JOBS_MODEL = 4

    def objective(trial: optuna.Trial, params: dict, X, y):
        temp = params.copy()
        temp.update(
            {
                "subsample": trial.suggest_float("subsample", 0.2, 1),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 5e-3, 5e-2, log=True
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.01, 0.5, log=True
                ),
                "feature_fraction_bynode": trial.suggest_float(
                    "feature_fraction_bynode", 0.05, 1, log=True
                ),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 35),
                "num_leaves": trial.suggest_int("num_leaves", 20, 45),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.00001, 0.1, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.00001, 0.1, log=True),
            }
        )

        scores = []
        # Outer split for test set
        outer_kf = KFold(n_splits=10, shuffle=True, random_state=0)
        for train_idx, test_idx in outer_kf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train model, report test performance
            model = lgb.LGBMRegressor(**temp, n_jobs=min(n_jobs, _N_JOBS_MODEL))
            model.fit(X_train, y_train, eval_metric="rmse")

            y_pred = model.predict(X_test)
            fold_rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
            scores.append(fold_rmse)
            trial.set_user_attr("best_iteration", model.best_iteration_)

        return np.median(scores)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": n_estimators,
        "verbosity": -1,
        "importance_type": "gain",
    }

    study = optuna.create_study(direction="minimize")
    objective_func = partial(objective, params=params, X=X, y=y)

    study.optimize(
        objective_func,
        n_trials=n_trials,
        n_jobs=max(1, n_jobs // _N_JOBS_MODEL),
        show_progress_bar=True,
    )
    params.update(study.best_params)

    return lgb.LGBMRegressor(**params)
