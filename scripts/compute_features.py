from tqdm import tqdm
from pathlib import Path
import os
from Bio import SeqIO
from aldiscore.prediction.extractor import FeatureExtractor
from aldiscore import ROOT
import pandas as pd
import multiprocessing as mp
from functools import partial
import datetime

_DONE = "DONE"


def get_dirs(dir):
    return list(filter(lambda name: os.path.isdir(dir / name), os.listdir(dir)))


def process_dataset(dataset: str, source: str, data_dir: Path, queue: mp.Queue):
    track_perf = queue is not None
    feats_dict = {}
    seqs = list(
        SeqIO.parse(data_dir / source / dataset / "sequences.fasta", format="fasta")
    )
    extractor = FeatureExtractor(seqs, track_perf=track_perf)
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
    log_path = ROOT / "logs" / "perf" / log_file
    while True:
        msg = queue.get()  # Read from the queue
        done = msg == _DONE
        if done:
            if len(perf_dicts) > 0:
                counter = 0
        else:
            source, dataset, perf_dict = msg
            perf_dicts[(source, dataset)] = perf_dict
            counter += 1

        if counter % 500 == 0:
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


track_perf = True
if __name__ == "__main__":

    data_dir = Path("/hits/fast/cme/bodynems/data/paper")
    sources = os.listdir(data_dir)
    # sources = ["ox"]
    print(f"Computing for sources: {sources}")

    print("Detected CPUs:", mp.cpu_count())
    cpu_counts = [min(100, mp.cpu_count() - 2)] * len(sources)
    if "treebase_v1" in sources:
        cpu_counts[sources.index("treebase_v1")] = mp.cpu_count() - 2

    manager = mp.Manager()
    queue = manager.Queue() if track_perf else None
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
        feat_df.to_parquet(data_dir / source / "features.parquet")

    queue.put(_DONE)
