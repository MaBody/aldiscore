import os
from pathlib import Path
import numpy as np
import pandas as pd
from aldiscore.prediction import utils
from aldiscore import ROOT, RSTATE
import lightgbm as lgb
from sklearn.model_selection import RepeatedKFold
from aldiscore.prediction.predictor import DifficultyPredictor
import psutil

data_dir = Path("/hits/fast/cme/bodynems/data/paper")
# Excluding non-relevant features
feat_df, drop_df, labels = utils.load_features(
    data_dir,
    exclude_features=["is_dna", "num_seqs", "seq_length", "10-mer_js", "13-mer_js"],
)

print(feat_df.shape)
print(drop_df.shape)
print(labels.shape)

X = feat_df
y = labels

n_jobs = psutil.cpu_count() - 4
print(f"Using {n_jobs} cores.")
# For 10 folds
rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
results = []
for train_idx, test_idx in rkf.split(feat_df, labels):
    X_train = feat_df.iloc[train_idx]
    y_train = labels.iloc[train_idx]
    X_test = feat_df.iloc[test_idx]
    y_test = labels.iloc[test_idx]
    # Run optuna search
    model = utils.optuna_search(
        X_train,
        y_train,
        n_trials=100,
        n_estimators=1200,
        n_jobs=n_jobs,
    )
    # train model
    model.fit(X_train, y_train)

    # report performance on held-out fold (RMSE; R^2, MAE)
    out = utils.compute_metrics(model, X_test, y_test)

    out = pd.concat([out, pd.DataFrame([model.get_params()])], axis=1)
    results.append(out)

result_df = pd.concat(results, axis=0, ignore_index=True)
result_df = result_df.sort_values("RMSE", ignore_index=True)
print(result_df.iloc[:4, :4])

out_path = (
    ROOT.parent
    / "logs"
    / "reporting"
    / f"report_{np.random.randint(100000,999999)}.parquet"
)
result_df.to_parquet(out_path)

# Train and save final model
best_params = dict(result_df.loc[0, ~result_df.columns.isin(out.columns)])
final_model = lgb.LGBMRegressor(**best_params)

final_model.fit(feat_df, labels)

predictor = DifficultyPredictor(final_model.booster_)
predictor.save("v1.1.txt")

print(out_path)
