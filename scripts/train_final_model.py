import os
from pathlib import Path
import numpy as np
import pandas as pd
from aldiscore.prediction import utils
from aldiscore import ROOT, RSTATE
from aldiscore.prediction.predictor import DifficultyPredictor

data_dir = Path("/hits/fast/cme/bodynems/data/paper")
feat_df, drop_df, label_df = utils.load_features(
    data_dir,
    exclude_features=["is_dna", "num_seqs", "seq_length", "10-mer_js", "13-mer_js"],
)

print(feat_df.shape)
print(drop_df.shape)
print(label_df.shape)

X = feat_df
y = label_df.iloc[:, 0]


model = utils.optuna_search(
    X,
    y,
    early_stopping=25,
    n_trials=500,
    n_estimators=1500,
    n_jobs=-1,
)
# train model
model.fit(X, y)

# save model
predictor = DifficultyPredictor(model.booster_)
predictor.save("v1.1.txt")
