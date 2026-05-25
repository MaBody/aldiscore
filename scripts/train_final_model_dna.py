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

data_dir = Path("/hits/fast/cme/luciamf/msa_difficulty/alignment-project/paper")
new_data_dir = Path("/hits/fast/cme/luciamf/msa_difficulty/new_data/") # new datasets
# Load features, excluding auxiliary/deprecated
feat_df1, drop_df1, labels1  = utils.load_features(
    data_dir,
    exclude_features=["is_dna", "num_seqs", "seq_length", "10-mer_js", "13-mer_js"],
    include_sources =  ["treebase_v1"],# "bali2dna"],
    # exclude_sources=["formatt_homstrad", "formatt_sabmark", "bali3", "prefab4", "ox", "sabre", "arthropod", "bali2dnaf", "bralibase_k5", "bralibase_k7", "bralibase_k15", "bali2dnaf"], # aa datasets
    data_type="DNA",
)


feat_df2, drop_df2, labels2  = utils.load_features(
    new_data_dir,
    exclude_features=["is_dna", "num_seqs", "seq_length", "10-mer_js", "13-mer_js"],
    include_sources = ["balibase_mdsa_all", "lanfear", "oxbench_mdsa_all", "smart_mdsa_all"] , # "balibase_mdsa_100s", "oxbench_mdsa_100s", "smart_mdsa_100s"],
    # exclude_sources=[], #, "bali2dnaf"], # aa datasets
    data_type="DNA",
)


feat_df = pd.concat([feat_df1, feat_df2], ignore_index=True)
drop_df = pd.concat([drop_df1, drop_df2], ignore_index=True)
labels  = pd.concat([labels1, labels2], ignore_index=True)

print(feat_df.shape)
print(drop_df.shape)
print(labels.shape)


metric_names = ["RMSE", "MAE", "R^2", "CORR"]
report_path = "/hits/fast/cme/luciamf/msa_difficulty/alignment-project/aldiscore/logs/reporting/report_v1.0_dna.parquet"

# Load performance report data
report_df = pd.read_parquet(report_path)
# best_params = dict(report_df.loc[0, ~report_df.columns.isin(metric_names)])
row = report_df.loc[0, ~report_df.columns.isin(metric_names)]

best_params = {
    k: (None if pd.isna(v) else v)
    for k, v in row.items()
}

# remove parameters you should not use
best_params.pop("class_weight", None)
best_params.pop("silent", None)

# fix random_state if missing
if best_params.get("random_state") is None:
    best_params["random_state"] = 53
    
print(best_params)


# Train and save final model
final_model = lgb.LGBMRegressor(**best_params)

final_model.fit(feat_df, labels)

predictor = DifficultyPredictor(final_model.booster_)
predictor.save("/hits/fast/cme/luciamf/msa_difficulty/alignment-project/aldiscore/aldiscore/models/v1.0_dna.txt")
