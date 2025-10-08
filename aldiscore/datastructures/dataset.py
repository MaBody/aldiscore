"""
Dataset module for handling collections of unaligned biological sequences.
Provides a Dataset class for storing, sorting, and inferring data types of sequence records.
"""

from Bio.SeqRecord import SeqRecord
from aldiscore.datastructures import utils
from aldiscore.enums.enums import DataTypeEnum
from typing import List


class Dataset:
    """
    Represents a collection of unaligned biological sequences.

    This class stores a list of sequence records, optionally sorts them,
    and infers the data type if not provided. It also provides access to
    sequence lengths and the original sorting indices.

    Attributes
    ----------
    records : list[SeqRecord]
        List of unaligned sequence records.
    data_type : DataTypeEnum
        Type of biological data (e.g., DNA, RNA, protein).
    _sorted : bool
        Whether the records have been sorted.
    _sort_idxs : list[int]
        Indices representing the sorting order (if sorted).
    _sequence_lengths : list[int]
        Lengths of each sequence in the dataset.
    """

    def __init__(
        self,
        records: List[SeqRecord],
        data_type: DataTypeEnum = None,
        sort_sequences: bool = True,
    ):
        """
        Initialize a Dataset instance.

        Parameters
        ----------
        records : list[SeqRecord]
            List of unaligned sequence records.
        data_type : DataTypeEnum, optional
            Type of biological data. If None, inferred automatically.
        sort_sequences : bool, default=True
            Whether to sort the sequences by a stable order.
        """
        # Apply stable sorting to the sequences
        self._sorted = sort_sequences
        if sort_sequences:
            self._sort_idxs = utils.argsort_seq_order(records)
            self.records = [records[idx] for idx in self._sort_idxs]
        else:
            self.records = records

        # If data_type is not provided, infer via heuristic
        if data_type is None:
            data_type = utils.infer_data_type(self.records)
        self.data_type = data_type

        self._sequence_lengths = [len(seq.seq) for seq in self.records]
