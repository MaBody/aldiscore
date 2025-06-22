"""
Ensemble module for handling collections of replicate sequence alignments.
Provides an Ensemble class for storing, referencing, and accessing alignment sets and their associated datasets.
"""

from aldiscore.datastructures.alignment import Alignment
from aldiscore.datastructures.dataset import Dataset


class Ensemble:
    """
    Represents a collection of replicate alignments for the same set of sequences.

    Stores a list of Alignment objects and the associated Dataset. Provides access to
    the data type and ensures the dataset is consistent with the alignments.

    Attributes
    ----------
    alignments : list[Alignment]
        List of replicate alignment objects.
    dataset : Dataset
        Dataset associated with the alignments.
    data_type : DataTypeEnum
        Type of biological data (e.g., DNA, RNA, protein).
    """

    def __init__(self, alignments: list[Alignment], dataset: Dataset = None):
        """
        Initialize an Ensemble instance.

        Parameters
        ----------
        alignments : list[Alignment]
            List of replicate alignment objects.
        dataset : Dataset, optional
            Dataset associated with the alignments. If None, inferred from the first alignment.
        """
        self.alignments = alignments

        if dataset is None:
            self.dataset = alignments[0].get_dataset()
        else:
            self.dataset = dataset

        self.data_type = self.dataset.data_type
