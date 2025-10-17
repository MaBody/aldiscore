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
# Load features, excluding auxiliary/deprecated
feat_df, drop_df, labels = utils.load_features(
    data_dir,
    exclude_features=["is_dna", "num_seqs", "seq_length", "10-mer_js", "13-mer_js"],
)


metric_names = ["RMSE", "MAE", "R^2", "CORR"]
report_path = ROOT.parent / "logs" / "reporting" / f"report_884441.parquet"

# Load performance report data
report_df = pd.read_parquet(report_path)
best_params = dict(report_df.loc[0, ~report_df.columns.isin(metric_names)])
print(best_params)

# Train and save final model
final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(feat_df, labels)

predictor = DifficultyPredictor(final_model.booster_)
predictor.save("v1.1.txt")
