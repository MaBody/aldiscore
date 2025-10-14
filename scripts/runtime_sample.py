# %%
from pathlib import Path
import os
import pandas as pd
from aldiscore import ROOT
import tempfile
from aldiscore.prediction.predictor import DifficultyPredictor
from aldiscore.scoring.pairwise import DPosDistance
from aldiscore.datastructures.ensemble import Ensemble
from time import perf_counter
import shutil
import multiprocessing as mp
from tqdm import tqdm

DATA_DIR = Path("/hits/fast/cme/bodynems/data/paper")
OUT_DIR = Path("/hits/fast/cme/bodynems/data/benchmarking")
sample = pd.read_parquet(ROOT.parent / "logs" / "misc" / "runtime_sample.parquet")
sample


def process_heuristic(row):
    source, dataset = row
    ens_dir = DATA_DIR / source / dataset / "ensemble"
    t1 = perf_counter()
    ensemble = Ensemble.load(ens_dir)
    DPosDistance().compute(ensemble, drop_gaps=False)
    t2 = perf_counter()
    return {(source, dataset): t2 - t1}

    # Compute heuristic for seq_path (with temp folder)


def process_prediction(row):
    source, dataset = row
    seq_file = DATA_DIR / source / dataset / "sequences.fasta"
    t1 = perf_counter()
    DifficultyPredictor().predict(seq_file, drop_gaps=False)
    t2 = perf_counter()
    return {(source, dataset): t2 - t1}


if __name__ == "__main__":
    perf_dict = {}
    n_cores = 70

    # Process heuristics
    with mp.Pool(n_cores) as pool:
        rows = map(lambda r: r[1].to_list(), sample.iterrows())
        for out in tqdm(pool.imap_unordered(process_heuristic, rows)):
            perf_dict.update(out)

    perf_df = pd.DataFrame(
        perf_dict.values(),
        index=perf_dict.keys(),
        columns=["Time (s)"],
    )
    perf_df.index.names = ["source", "dataset"]
    perf_df.to_parquet(ROOT.parent / "logs" / "misc" / "heuristic_time.parquet")

    # Process predictions
    perf_dict = {}
    with mp.Pool(n_cores) as pool:
        rows = map(lambda r: r[1].to_list(), sample.iterrows())
        for out in tqdm(pool.imap_unordered(process_prediction, rows)):
            perf_dict.update(out)

    perf_df = pd.DataFrame(
        perf_dict.values(),
        index=perf_dict.keys(),
        columns=["Time (s)"],
    )
    perf_df.index.names = ["source", "dataset"]
    perf_df.to_parquet(ROOT.parent / "logs" / "misc" / "prediction_time.parquet")


# %%
