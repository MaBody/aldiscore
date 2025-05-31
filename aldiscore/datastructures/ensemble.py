from aldiscore.datastructures.alignment import Alignment
from aldiscore.datastructures.dataset import Dataset


class Ensemble:
    """
    Class encapsulating a set of replicate alignments for the same set of sequences.
    This class provides methods for computing features on this set of alignments.

    Parameters
    ----------
    ensemble : list[Alignment]
        List of replicate alignments (subclass of PyPythia `MSA`).
    dataset : Dataset
        The dataset on which the ensemble is computed.

    Attributes
    ----------
    ensemble : list[Alignment]
        List of replicate alignments (subclass of PyPythia `MSA`).
    dataset : Dataset
        The dataset on which the ensemble is computed.
    """

    def __init__(self, alignments: list[Alignment], dataset: Dataset = None):
        self.alignments = alignments

        if dataset is None:
            self.dataset = alignments[0].get_dataset()
        else:
            self.dataset = dataset

        self.data_type = self.dataset.data_type
