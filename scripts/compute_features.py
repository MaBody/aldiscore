from tqdm import tqdm
from pathlib import Path
import os
from Bio import SeqIO
from aldiscore.prediction.extractor import FeatureExtractor
import pandas as pd
import multiprocessing
from functools import partial


def get_dirs(dir):
    return list(filter(lambda name: os.path.isdir(dir / name), os.listdir(dir)))


def process_dataset(dataset: str, data_dir=None, source=None):
    feats_dict = {}
    seqs = list(
        SeqIO.parse(data_dir / source / dataset / "sequences.fasta", format="fasta")
    )
    feats = FeatureExtractor(seqs).compute()
    feats_dict[(source, dataset)] = feats
    return feats_dict


if __name__ == "__main__":

    data_dir = Path("/hits/fast/cme/bodynems/data/paper")
    sources = os.listdir(data_dir)
    # sources = ["ox"]
    print(f"Computing for sources: {sources}")

    cpu_count = multiprocessing.cpu_count()
    print("CPUs:", cpu_count)

    for source in sources:
        datasets = get_dirs(data_dir / source)
        feats_dict = {}

        process_func = partial(process_dataset, data_dir=data_dir, source=source)
        with multiprocessing.Pool(cpu_count - 2) as pool:
            print(f"[SOURCE] {source}")
            for out in tqdm(pool.imap_unordered(process_func, datasets)):
                feats_dict.update(out)

        feat_df = pd.concat(feats_dict.values(), axis=0, ignore_index=True)
        feat_df.index = pd.MultiIndex.from_tuples(
            feats_dict.keys(), names=["source", "dataset"]
        )
        feat_df.to_parquet(data_dir / source / "features.parquet")
