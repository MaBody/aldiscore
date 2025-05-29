import pytest
import dotenv
import numpy as np
import pathlib
import itertools
import features.pairwise
import aldiscore.enums.enums
from aldiscore.datastructures.dataset import Dataset
from aldiscore.datastructures.ensemble import Ensemble
from aldiscore.datastructures.alignment import Alignment
import os
from features.aligned import EnsembleFeatureExtractor

dotenv.load_dotenv()
ROOT = pathlib.Path(".")
TEST_DATA_DIR = ROOT / "tests" / "data"

dataset = Dataset(TEST_DATA_DIR / "sequences.fasta")
ensemble = []
for msa_file in os.listdir(TEST_DATA_DIR):
    if msa_file.startswith("protein"):
        alignment = Alignment(
            TEST_DATA_DIR / msa_file,
            data_type=dataset.data_type,
            file_format=FileFormat.FASTA,
        )
        ensemble.append(alignment)
ensemble = Ensemble(ensemble, dataset)


metal = pathlib.Path("/hits/fast/cme/bodynems/tools/metal")
extractor = EnsembleFeatureExtractor(
    ensemble, dataset, None, None, None, metal, compute_distance_matrix=True, threads=1
)


def test_homology_set_metrics():
    ssp_new = features.pairwise.SSPDistance()
    seq_new = features.pairwise.HomologySeqDistance()
    pos_new = features.pairwise.HomologyPosDistance()
    metrics = {metric.name: metric for metric in [ssp_new, seq_new, pos_new]}
    dists_new_dict = {name: [] for name in metrics}
    for alignment_x, alignment_y in itertools.combinations(ensemble.ensemble, r=2):
        for name, metric in metrics.items():
            dist = metric.compute(alignment_x, alignment_y)
            dists_new_dict[name].append(dist)

    dists_ref_dict = extractor._metal_stats()
    for name in dists_new_dict:
        dists_new = np.array(dists_new_dict[name])
        dists_ref = np.array(dists_ref_dict[name + "_metal"])
        np.testing.assert_array_almost_equal(
            dists_new, dists_ref, decimal=3, err_msg=name
        )
