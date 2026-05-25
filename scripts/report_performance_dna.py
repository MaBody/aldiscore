import os
from pathlib import Path
import numpy as np
import pandas as pd
from aldiscore.prediction import utils
from aldiscore import ROOT, RSTATE
import lightgbm as lgb
from aldiscore.prediction.predictor import DifficultyPredictor
import psutil
import sys

data_dir = Path("/hits/fast/cme/luciamf/msa_difficulty/alignment-project/paper")
new_data_dir = Path("/hits/fast/cme/luciamf/msa_difficulty/new_data/") # new datasets
# Load features, excluding auxiliary/deprecated
feat_df1, drop_df1, labels1  = utils.load_features(
    data_dir,
    exclude_features=["is_dna", "num_seqs", "seq_length", "10-mer_js", "13-mer_js"],
    include_sources =  ["treebase_v1", "bali2dna"],
    #exclude_sources=["invalid_datasets, formatt_homstrad", "formatt_sabmark", "bali3", "prefab4", "ox", "sabre", "arthropod", "bali2dnaf", "bralibase_k5", "bralibase_k7", "bralibase_k15", "bali2dnaf"], # aa datasets
    data_type="DNA",
)


feat_df2, drop_df2, labels2  = utils.load_features(
    new_data_dir,
    exclude_features=["is_dna", "num_seqs", "seq_length", "10-mer_js", "13-mer_js"],
    include_sources = ["balibase_mdsa_all", "lanfear", "oxbench_mdsa_all", "smart_mdsa_all", "balibase_mdsa_100s", "oxbench_mdsa_100s", "smart_mdsa_100s"],
    # exclude_sources=[], #, "bali2dnaf"], # aa datasets
    data_type="DNA",
)


feat_df = pd.concat([feat_df1, feat_df2], ignore_index=True)
drop_df = pd.concat([drop_df1, drop_df2], ignore_index=True)
labels  = pd.concat([labels1, labels2], ignore_index=True)

print(feat_df.shape)
print(drop_df.shape)
print(labels.shape)

X = feat_df
y = labels

n_jobs = 36 #psutil.cpu_count() - 4
print(f"Using {n_jobs} cores.")
# For 10 folds
results = []
rkf = utils.RepeatedStratifiedKFoldReg(n_splits=10, n_repeats=10,n_bins=5, random_state=0)
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

out_path = "/hits/fast/cme/luciamf/msa_difficulty/alignment-project/aldiscore/logs/reporting/report_v1.0_dna.parquet" 
result_df.to_parquet(out_path)

# Train and save final model
best_params = dict(result_df.loc[0, ~result_df.columns.isin(out.columns)])
final_model = lgb.LGBMRegressor(**best_params)

final_model.fit(feat_df, labels)

predictor = DifficultyPredictor(final_model.booster_)
predictor.save("v0.0_dna.txt")

print(out_path)
