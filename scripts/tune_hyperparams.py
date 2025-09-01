import optuna
import lightgbm as lgb
from pathlib import Path
from aldiscore.prediction import utils
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from aldiscore import ROOT
from functools import partial
import os
import pandas as pd
import numpy as np

optuna.logging.set_verbosity(optuna.logging.ERROR)


def rmse(model, X, y):
    y_pred = model.predict(X)
    return ((y_pred - y) ** 2).mean() ** 0.5


def objective(trial, X_train, y_train, X_valid, y_valid):
    params = {
        "n_jobs": 1,
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 1000,
        "verbosity": -1,
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    kf = KFold(n_splits=5, shuffle=True, random_state=RSTATE)
    scores = cross_val_score(model, X_valid, y_valid, scoring=rmse, cv=kf)

    return max(np.mean(scores), np.median(scores))
    # return rmse


RSTATE = 0
if __name__ == "__main__":
    data_dir = Path("/hits/fast/cme/bodynems/data/paper")
    save_dir = ROOT / "optuna"
    os.makedirs(save_dir, exist_ok=True)

    feat_df, drop_df, label_df = utils.load_features(
        data_dir,
        exclude_features=["is_dna", "num_seqs", "seq_length"],
    )
    print(feat_df.shape)
    print(drop_df.shape)
    print(label_df.shape)

    # Remove ':' chars from names (does not work with LightGBM)
    clean_feat_names = feat_df.columns.str.replace(":", ".").to_list()
    feat_df.columns = clean_feat_names

    train_idxs, test_idxs = train_test_split(
        feat_df.index.to_list(), test_size=0.2, random_state=RSTATE
    )
    test_idxs, valid_idxs = train_test_split(
        test_idxs, test_size=0.5, random_state=RSTATE
    )
    # print(len(train_idxs), len(test_idxs), len(valid_idxs))
    X_train = feat_df.loc[train_idxs]
    X_test = feat_df.loc[test_idxs]
    X_valid = feat_df.loc[valid_idxs]
    y_train = label_df.loc[train_idxs].iloc[:, 0]
    y_test = label_df.loc[test_idxs].iloc[:, 0]
    y_valid = label_df.loc[valid_idxs].iloc[:, 0]

    # # Create LightGBM datasets
    # train_data = lgb.Dataset(
    #     feat_df.loc[train_idxs],
    #     label=label_df.loc[train_idxs].to_numpy().ravel(),
    # )
    # test_data = lgb.Dataset(
    #     feat_df.loc[test_idxs],
    #     label=label_df.loc[test_idxs].to_numpy().ravel(),
    # )
    # valid_data = lgb.Dataset(
    #     feat_df.loc[valid_idxs],
    #     label=label_df.loc[valid_idxs].to_numpy().ravel(),
    # )

    # train_data = train_data.set_feature_name(clean_feat_names)
    # test_data = test_data.set_feature_name(clean_feat_names)
    # valid_data = valid_data.set_feature_name(clean_feat_names)

    study = optuna.create_study(direction="minimize")

    objective_func = partial(
        objective, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid
    )
    study.optimize(objective_func, n_trials=1000, n_jobs=-1, show_progress_bar=True)
    results = []
    for trial in study.trials:
        result = {}
        result["score"] = trial.value
        result.update(trial.params)
        results.append(result)

    results = pd.DataFrame(results)
    results = results.sort_values("score", ignore_index=True)
    results.to_parquet(save_dir / "trial_01.parquet")
    print(results.head())
