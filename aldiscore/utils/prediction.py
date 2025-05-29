# from dotenv import load_dotenv

# load_dotenv()

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor
from time import localtime, strftime
from utils.dataloading import load_balibase_results
from sklearn.model_selection import train_test_split
from functools import partial
from pathlib import Path
import itertools
import tqdm
from multiprocessing import Pool

RESULT_DIR = Path("/hits/fast/cme/bodynems/MSA_difficulty/results")
DATA_DIR = Path("/hits/fast/cme/bodynems/data")


def rmse(y, y_pred, **kwargs):
    return np.sqrt(((y - y_pred) ** 2 / len(y)).sum())


def mape(y, y_pred, eps=1e-6, **kwargs):
    return ((y - y_pred) / (y + eps)).abs().mean()


def mae(y, y_pred, **kwargs):
    return (y - y_pred).abs().mean()


def corr(y, y_pred, **kwargs):
    return np.corrcoef(y, y_pred)[0, 1]


METRICS = {"RMSE": rmse, "MAPE": mape, "MAE": mae, "Corr": corr}


# def compute_perf_metrics(y, y_pred, name="all", eps=1e-6):
#     results = {key: val(y, y_pred, eps=eps) for key, val in METRICS.items()}
#     results = pd.DataFrame(results, index=[name])
#     return results

#     # Define Features to use in some file
#     # Input: unaligned data, aligned data


# def prepare_data(unaligned_features, aligned_features):
#     # Returns cleaned X and y datasets, along with stratification groups
#     pass


# def hyperparameter_search(X):

#     # Optimism Corrected
#     model = LinearRegression()
#     model.fit(X, y)
#     apparent_performance = mean_squared_error(y, model.predict(X))

#     mse = make_scorer(mean_squared_error)
#     cv = cross_validate(model, X, y, cv=opt_cv, scoring=mse, return_train_score=True)
#     optimism = cv["test_score"] - cv["train_score"]
#     optimism_corrected = apparent_performance + optimism.mean()
#     print(f"Optimism Corrected: {optimism_corrected:.2f}")

#     # Compare against regular cv
#     cv = cross_validate(model, X, y, cv=10, scoring=mse)["test_score"].mean()
#     print(f"regular cv: {cv:.2f}")

#     # Compare against repeated cv
#     cv = cross_validate(
#         model, X, y, cv=RepeatedKFold(n_splits=10, n_repeats=100), scoring=mse
#     )["test_score"].mean()
#     print(f"repeated cv: {cv:.2f}")


# pd.qcut(y, q=10)


def get_ml_data(y_feat="mean_homology_pos_dist", source=["treebase"]):
    (
        unaligned_features,
        aligned_features,
        aligned_cols,
        reference_features,
        confusion_features,
        _,
        _,
    ) = load_balibase_results(source, with_reference=False, benchmarking_scores=False)

    X = unaligned_features.sort_values("dataset", ignore_index=True)
    X = X.drop(["benchmark"], axis=1)
    X["data_type"], data_type_mapping = pd.factorize(X["data_type"])
    print(data_type_mapping)
    y = (
        aligned_features.sort_values("dataset", ignore_index=True)
        .loc[aligned_features.tool == "all", ["dataset", y_feat]]
        .reset_index(drop=True)
    )
    # print((y.dataset == X.dataset).mean())
    y = y[y_feat]
    return (X, y)


def run_cv_job(X_train, y_train, args: dict, stratify_col: pd.Series = None):
    results = {}
    model = GradientBoostingRegressor(**args)
    cv_out = cross_validate(
        model,
        X_train,
        y_train,
        cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=0),
        scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"),
    )
    results[tuple(args.values())] = {key: np.mean(val) for key, val in cv_out.items()}
    print("Job done.")
    return results


def hyperarameter_search(X_train, y_train, save=True):

    params = {
        "n_estimators": [2000],
        "learning_rate": [0.1, 0.05, 0.02, 0.01],
        "max_depth": [2, 4, 8],
        "min_samples_leaf": [3, 5, 9, 17],
        "max_features": [1.0, 0.6, 0.3, 0.1],
        "loss": ["squared_error", "absolute_error", "huber"],
    }

    # params = {
    #     "max_depth": [2, 4],
    #     "min_samples_leaf": [3, 5],
    # }

    param_grid = list(itertools.product(*params.values()))
    param_grid = list(map(lambda vals: dict(zip(params, vals)), param_grid))

    results = Pool(64).map(
        partial(run_cv_job, X_train, y_train),
        param_grid,
    )

    result_df = {}
    for result in results:
        result_df.update(result)

    result_df = pd.DataFrame(result_df.values(), index=result_df.keys())
    result_df.index = result_df.index.set_names(list(params.keys()))

    return result_df


def learning_rate_search():
    pass


if __name__ == "__main__":
    X, y = get_ml_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=X["data_type"], random_state=0
    )
    result_df = hyperarameter_search(X_train[:], y_train[:])
    now = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    result_df.to_pickle(RESULT_DIR / "treebase" / "modeling" / f"{now}_crossval.pkl")
