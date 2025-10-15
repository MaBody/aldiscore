# %%
from pathlib import Path
import os
import pandas as pd
import tempfile
from aldiscore.prediction.predictor import DifficultyPredictor
from aldiscore.scoring.pairwise import DPosDistance
from aldiscore.datastructures.ensemble import Ensemble
from time import perf_counter
import shutil
import multiprocessing as mp
from tqdm import tqdm
import subprocess

DATA_DIR = Path("/hits/fast/cme/bodynems/data/paper")
ROOT = Path("/hits/fast/cme/bodynems/aldiscore/aldiscore")

sample = pd.read_parquet(ROOT.parent / "logs" / "misc" / "runtime_sample.parquet")
sample
cli_head = ["python", str(ROOT / "main.py")]


def process_heuristic(row):
    source, dataset = row
    ens_dir = DATA_DIR / source / dataset / "ensemble"
    t1 = perf_counter()
    out = subprocess.run(cli_head + ["heuristic", str(ens_dir)], capture_output=True)
    # ensemble = Ensemble.load(ens_dir)
    # score = DPosDistance().compute(ensemble)
    t2 = perf_counter()
    return {(source, dataset): t2 - t1}

    # Compute heuristic for seq_path (with temp folder)


def process_prediction(row):
    source, dataset = row
    seq_file = DATA_DIR / source / dataset / "sequences.fasta"
    t1 = perf_counter()
    out = subprocess.run(cli_head + ["predict", str(seq_file)], capture_output=True)
    # DifficultyPredictor().predict(seq_file, drop_gaps=False)
    t2 = perf_counter()
    return {(source, dataset): t2 - t1}


if __name__ == "__main__":
    perf_dict = {}
    n_cores = 40
    perf_heuristic = None
    perf_prediction = None
    rows = map(lambda r: r[1].to_list(), sample.iterrows())
    iterable = list(rows)[::-1]
    n = 5
    for i in range(n):
        # Process heuristics
        with mp.Pool(n_cores) as pool:
            for out in tqdm(pool.imap_unordered(process_heuristic, iterable)):
                perf_dict.update(out)

        perf_df = pd.DataFrame(
            perf_dict.values(),
            index=perf_dict.keys(),
            columns=["Time (s)"],
        )
        perf_df.index.names = ["source", "dataset"]
        if perf_heuristic is None:
            perf_heuristic = perf_df
        else:
            perf_heuristic += perf_df
        # perf_df.to_parquet(ROOT.parent / "logs" / "misc" / "heuristic_time.parquet")

        # Process predictions
        perf_dict = {}
        with mp.Pool(n_cores) as pool:
            for out in tqdm(pool.imap_unordered(process_prediction, iterable)):
                perf_dict.update(out)

        perf_df = pd.DataFrame(
            perf_dict.values(),
            index=perf_dict.keys(),
            columns=["Time (s)"],
        )
        perf_df.index.names = ["source", "dataset"]
        if perf_prediction is None:
            perf_prediction = perf_df
        else:
            perf_prediction += perf_df

    perf_heuristic /= n
    perf_prediction /= n

    perf_heuristic.to_parquet(ROOT.parent / "logs" / "misc" / "heuristic_time.parquet")
    perf_prediction.to_parquet(
        ROOT.parent / "logs" / "misc" / "prediction_time.parquet"
    )
    perf_prediction
