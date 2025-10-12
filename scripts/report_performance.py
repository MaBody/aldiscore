import os
from pathlib import Path
import numpy as np
import pandas as pd
from aldiscore.prediction import utils
from aldiscore import ROOT, RSTATE


data_dir = Path("/hits/fast/cme/bodynems/data/paper")
feat_df, drop_df, label_df = utils.load_features(
    data_dir,
    exclude_features=["is_dna", "num_seqs", "seq_length", "10-mer_js", "13-mer_js"],
)

print(feat_df.shape)
print(drop_df.shape)
print(label_df.shape)


from sklearn.model_selection import RepeatedKFold

# For 10 folds
rkf = RepeatedKFold(n_splits=10, n_repeats=2, random_state=0)
results = []
for train_idx, test_idx in rkf.split(feat_df, label_df.iloc[:, 0]):
    X_train = feat_df.iloc[train_idx]
    y_train = label_df.iloc[train_idx, 0]
    X_test = feat_df.iloc[test_idx]
    y_test = label_df.iloc[test_idx, 0]
    # Run optuna search
    model = utils.optuna_search(
        X_train,
        y_train,
        early_stopping=25,
        n_trials=150,
        n_estimators=1500,
        n_jobs=-1,
    )
    # train model
    model.fit(X_train, y_train)

    # report performance on held-out fold (RMSE; R^2, MAE)
    out = utils.compute_metrics(model, X_test, y_test)

    out = pd.concat([out, pd.DataFrame([model.get_params()])], axis=1)

    results.append(out)

result_df = pd.concat(results, axis=0, ignore_index=True)
out_path = (
    ROOT.parent
    / "logs"
    / "reporting"
    / f"report_{np.random.randint(100000,999999)}.parquet"
)
result_df.to_parquet(out_path)
print(out_path)
