from tqdm import tqdm
from pathlib import Path
import os
from Bio import SeqIO
from aldiscore.prediction.extractor import FeatureExtractor
from aldiscore.constants.constants import DNA_CHARS, AA_CHARS, GAP_CHAR
from aldiscore import ROOT
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
import datetime

_DONE = "DONE"
_TRACK_PERF = True


def get_dirs(dir):
    return list(filter(lambda name: os.path.isdir(dir / name), os.listdir(dir)))


def infer_data_type(records) -> str:
    """
    Composition heuristic that labelled the v1.0 training data (`is_dna` feature).
    No longer part of the package; kept here so features stay reproducible,
    since sources like treebase_v1 mix DNA and AA datasets.
    """
    ungapped = None
    for seq in records:
        ungapped = str(seq.seq).replace(GAP_CHAR, "")[:1000].upper()
        if len(ungapped) >= 100:
            break
    char_counts = pd.Series(list(ungapped)).value_counts()
    unique_chars = char_counts.index.to_series()

    dna_included = np.isin(DNA_CHARS, unique_chars)
    protein_included = np.isin(AA_CHARS, unique_chars)

    protein_missing = AA_CHARS[~protein_included]
    missing_counts = pd.Series(0, index=protein_missing)
    char_counts = pd.concat((char_counts, missing_counts))

    # If all DNA chars are included and their cumulative share is above 90%: DNA!
    if dna_included.all() & (char_counts[DNA_CHARS].sum() / len(ungapped) >= 0.9):
        return "DNA"
    # If the cumulative share of protein chars is above 90%: Protein!
    elif char_counts[AA_CHARS].sum() / len(ungapped) >= 0.9:
        return "AA"
    else:
        raise AssertionError(
            "Unknown alphabet detected for string '{}'".format(ungapped)
        )


def process_dataset(dataset: str, source: str, data_dir: Path, queue: mp.Queue):
    track_perf = queue is not None
    feats_dict = {}
    seqs = list(
        SeqIO.parse(data_dir / source / dataset / "sequences.fasta", format="fasta")
    )
    psa_config = {"MAX_PSA_COUNT": 300}
    extractor = FeatureExtractor(
        seqs,
        data_type=infer_data_type(seqs),
        psa_config=psa_config,
        track_perf=track_perf,
        validate="warn",
    )
    feats = extractor.compute()

    if track_perf:
        queue.put((source, dataset, extractor._perf_dict))

    feats_dict[(source, dataset)] = feats
    return feats_dict


def process_queue(queue: mp.Queue):
    """Read from the queue; this spawns as a separate Process"""
    counter = 0
    perf_dicts = {}
    log_file = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".parquet"
    log_path = "/hits/fast/cme/luciamf/msa_difficulty/alignment-project/aldiscore/logs/perf/" + log_file
    # log_path = ROOT.parent / "logs" / "perf" / log_file
    
    while True:
        msg = queue.get()  # Read from the queue
        done = msg == _DONE
        if done:
            # If done and elements unsaved, set counter to 0 to save them
            if len(perf_dicts) > 0:
                counter = 0
        else:
            source, dataset, perf_dict = msg
            perf_dicts[(source, dataset)] = perf_dict
            counter += 1

        if counter % 500 == 0:
            # Store every 500 completions to save bandwidth
            perf_df = pd.DataFrame(perf_dicts.values())
            perf_df.index = pd.MultiIndex.from_tuples(
                perf_dicts.keys(), names=["source", "dataset"]
            )
            if os.path.exists(log_path):
                perf_df_pre = pd.read_parquet(log_path)
                perf_df = pd.concat((perf_df_pre, perf_df), axis=0)
            perf_df.to_parquet(log_path)
            perf_dicts.clear()

        if done:
            print("Queue is done...")
            break


if __name__ == "__main__":

    data_dir = Path("/hits/fast/cme/luciamf/msa_difficulty/alignment-project/paper/") # og_datasets
    #data_dir = Path("/hits/fast/cme/luciamf/msa_difficulty/pandit/ensembles") # pandit datasets
    # data_dir = Path("/hits/fast/cme/luciamf/msa_difficulty/new_data/") # new datasets
    # sources = ["aa", "dna"] # pandit
    # sources = ["balibase_mdsa_all", "lanfear", "oxbench_mdsa_all", "smart_mdsa_all", "balibase_mdsa_100s", "oxbench_mdsa_100s", "smart_mdsa_100s"]
    # faltan PREFAB LOL
    # sources.remove("treebase_v1")
    # sources = ["treebase_v1"] + sources
    # sources.remove("bralibase_k5")
    # sources.remove("bralibase_k7")
    # sources.remove("bralibase_k15")
    # sources.remove("bali2dna")
    # sources.remove("bali2dnaf") # dna datasets
    sources = ["bali2dna", # dna
               "treebase_v1", # both
               "arthropod","formatt_homstrad_renamed","formatt_sabmark_renamed", "prefab4", "bali3", "ox", "sabre"] # aa
    
    print(f"Computing for sources: {sources}")

    print("Detected CPUs:", mp.cpu_count())
    cpu_counts = [min(100, mp.cpu_count() - 10)] * len(sources)
    
    #if "treebase_v1" in sources:
       # cpu_counts[sources.index("treebase_v1")] = 30 # mp.cpu_count() - 10

    manager = mp.Manager()
    queue = manager.Queue() if _TRACK_PERF else None
    if queue is not None:
        mp.Process(target=process_queue, args=(queue,)).start()

    for source, cpu_count in zip(sources, cpu_counts):
        datasets = get_dirs(data_dir / source)
        feats_dict = {}

        process_func = partial(
            process_dataset, data_dir=data_dir, source=source, queue=queue
        )
        with manager.Pool(cpu_count) as pool:
            print(f"[SOURCE] {source} (CPUs: {cpu_count})")
            for out in tqdm(pool.imap_unordered(process_func, datasets)):
                feats_dict.update(out)

        feat_df = pd.concat(feats_dict.values(), axis=0, ignore_index=True)
        feat_df.index = pd.MultiIndex.from_tuples(
            feats_dict.keys(), names=["source", "dataset"]
        )
        feat_df.to_parquet(data_dir / source / "features.parquet", engine="fastparquet")

    queue.put(_DONE)
