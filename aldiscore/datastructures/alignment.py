import pathlib
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from aldiscore.enums.enums import DataTypeEnum
from aldiscore.datastructures.dataset import Dataset
from aldiscore.constants.constants import GAP_CHAR
from aldiscore.datastructures import utils


class Alignment:
    """
    Represents a multiple sequence alignment (MSA) and provides utility methods for
    manipulating and exporting the alignment.

    Attributes:
        msa (MultipleSeqAlignment): The Biopython MSA object, optionally sorted.
        data_type (DataTypeEnum): The type of data (e.g., DNA, RNA, protein).
        key (str): Unique key for the alignment, used for caching.
        shape (tuple): Tuple of (number of sequences, alignment length).
        _sorted (bool): Whether the sequences are sorted.
        _sort_idxs (list): Indices used for sorting, if applicable.
    """

    def __init__(
        self,
        msa: MultipleSeqAlignment,
        data_type: DataTypeEnum = None,
        sort_sequences: bool = True,
        # name: str = None,
    ):
        """
        Initialize an Alignment object.

        Args:
            msa (MultipleSeqAlignment): The input MSA object.
            data_type (DataTypeEnum, optional): Data type of the sequences. If None, inferred automatically.
            sort_sequences (bool, optional): If True, sequences are sorted in a stable order.
        """
        # Applies a stable sorting to the sequences
        self._sorted = sort_sequences
        if sort_sequences:
            self._sort_idxs = utils.argsort_seq_order(msa)
            self.msa = MultipleSeqAlignment([msa[idx] for idx in self._sort_idxs])
        else:
            self.msa = msa

        # If data_type is not provided, infer via heuristic
        if data_type is None:
            data_type = utils.infer_data_type(self.msa)
        self.data_type = data_type

        # Compute a unique key for the alignment (used in caching intermediate results)
        self.key = utils.get_unique_key(msa)

        # Set shape attribute for convience
        self.shape = (len(self.msa), self.msa.get_alignment_length())

    def save_to_fasta(self, out_file: pathlib.Path, no_gaps: bool = False):
        """
        Save the alignment to a FASTA file.

        Args:
            out_file (pathlib.Path): Output file path.
            no_gaps (bool, optional): If True, gaps are removed from sequences before saving.
        """
        with open(out_file, "w") as outfile:
            for sequence in self.msa:
                outfile.write(f">{sequence.id}\n")
                if no_gaps:
                    outfile.write(str(sequence.seq).replace(GAP_CHAR, ""))
                else:
                    outfile.write(str(sequence.seq))
                outfile.write("\n")

    def get_dataset(self):
        """
        Convert the alignment to a Dataset object with all gaps removed.

        Returns:
            Dataset: Dataset object containing ungapped sequences.
        """
        ungapped_records = []
        for record in self.msa:
            rec = SeqRecord(Seq(str(record.seq).replace(GAP_CHAR, "")), id=record.id)
            ungapped_records.append(rec)
        dataset = Dataset(ungapped_records, self.data_type, self._sorted)
        return dataset
