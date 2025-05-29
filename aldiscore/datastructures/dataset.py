import pathlib
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Align import MultipleSeqAlignment
import datastructures.utils
import numpy as np
from enums.enums import DataTypeEnum


class Dataset:
    """
    Class structure encapsulating raw, unaligned sequences.
    This class provides methods for computing features on the unaligned sequences, as well as for generating MSAs using
    various different MSA tools.

    Parameters
    ----------
    sequences : pathlib.Path
        Filepath of the unaligned sequences in FASTA format.

    Attributes
    ----------
    sequences : pathlib.Path
        Filepath of the unaligned sequences in FASTA format.
    sequence_records : list[SeqIO.SeqRecord]
        List containing the unaligned sequences as `SeqIO.SeqRecord` objects.
    """

    def __init__(
        self,
        records: list[SeqRecord],
        data_type: DataTypeEnum = None,
        sort_sequences: bool = True,
    ):
        # Apply stable sorting to the sequences
        self._sorted = sort_sequences
        if sort_sequences:
            self._sort_idxs = datastructures.utils.argsort_seq_order(records)
            self.records = [records[idx] for idx in self._sort_idxs]
        else:
            self.records = records

        # If data_type is not provided, infer via heuristic
        if data_type is None:
            data_type = datastructures.utils.infer_data_type(self.records)
        self.data_type = data_type

        self._sequence_lengths = [len(seq.seq) for seq in self.sequences]
