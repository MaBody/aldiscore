import pytest
import numpy as np
import pathlib
from aldiscore.scoring import confusion
from aldiscore.datastructures import alignment, ensemble
import os

# from Bio.Align import MultipleSeqAlignment
from Bio.AlignIO import read

TEST_DATA_DIR = pathlib.Path.cwd() / "tests" / "data"


_alignments: list[alignment.Alignment] = []
for msa_file in os.listdir(TEST_DATA_DIR):
    if msa_file.startswith("protein"):
        _alignment = alignment.Alignment(
            read(TEST_DATA_DIR / msa_file, "fasta"),
            data_type="AA",
            sort_sequences=True,
        )
        _alignments.append(_alignment)
_dataset = _alignment.get_dataset()
_ensemble = ensemble.Ensemble(_alignments, _dataset)


def test_datastructures():
    print(print([a.shape for a in _alignments]))
    print(len(_dataset.records))
    print([len(seq) for seq in _dataset.records])
    metric = confusion.ConfusionSet(aggregate="site")
    score = metric.compute(_ensemble)
    print(score)
    metric = confusion.ConfusionEntropy(aggregate="site")
    score = metric.compute(_ensemble)
    print(score)
    metric = confusion.ConfusionDisplace(aggregate="site")
    score = metric.compute(_ensemble)
    print(score)


# metal = pathlib.Path("/hits/fast/cme/bodynems/tools/metal")
# extractor = EnsembleFeatureExtractor(
#     ensemble, dataset, None, None, None, metal, compute_distance_matrix=True, threads=1
# )


# def test_homology_set_metrics():
#     ssp_new = features.pairwise.SSPDistance()
#     seq_new = features.pairwise.HomologySeqDistance()
#     pos_new = features.pairwise.HomologyPosDistance()
#     metrics = {metric.name: metric for metric in [ssp_new, seq_new, pos_new]}
#     dists_new_dict = {name: [] for name in metrics}
#     for alignment_x, alignment_y in itertools.combinations(ensemble.ensemble, r=2):
#         for name, metric in metrics.items():
#             dist = metric.compute(alignment_x, alignment_y)
#             dists_new_dict[name].append(dist)

#     dists_ref_dict = extractor._metal_stats()
#     for name in dists_new_dict:
#         dists_new = np.array(dists_new_dict[name])
#         dists_ref = np.array(dists_ref_dict[name + "_metal"])
#         np.testing.assert_array_almost_equal(
#             dists_new, dists_ref, decimal=3, err_msg=name
#         )


if __name__ == "__main__":
    test_datastructures()
