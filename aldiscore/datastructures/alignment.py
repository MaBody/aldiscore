import pathlib
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from aldiscore.enums.enums import DataTypeEnum
from aldiscore.datastructures.dataset import Dataset
from aldiscore.constants.constants import GAP_CHAR
from aldiscore.datastructures import utils


class Alignment:
    """Class structure for Multiple Sequence Alignment statistics. Applies stable ordering to sequences.

    Args
    ----
        msa_file (FilePath): Path to the MSA file the statistics should be computed for. The file must be either in "fasta" or "phylip" format.
        msa_name (str, optional): Better readable name for the MSA. If not set, the name of the file is used.
        file_format (FileFormat, optional): If not set, Pythia attempts to autodetect the file format.

    Attributes
    ----------
        msa_file (FilePath): Path to the corresponding MSA file.
        msa_name (str): Name of the MSA. Can be either set or is inferred automatically based on the msa_file.
        data_type (DataType): Data type of the MSA.
        file_format (FileFormat): File format of the MSA.
        msa (MultipleSeqAlignment): Biopython MultipleSeqAlignment object for the given msa_file.
        _tool:str
            The name of the tool that produced the alignment (internal use only).

    Raises
    ------
        ValueError: If the file format of the given MSA is not FASTA or PHYLIP.
        ValueError: If the data type of the given MSA cannot be inferred.
    """

    def __init__(
        self,
        msa: MultipleSeqAlignment,
        data_type: DataTypeEnum = None,
        sort_sequences: bool = True,
        # name: str = None,
    ):
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
        with open(out_file, "w") as outfile:
            for sequence in self.msa:
                outfile.write(f">{sequence.id}\n")
                if no_gaps:
                    outfile.write(str(sequence.seq).replace(GAP_CHAR, ""))
                else:
                    outfile.write(str(sequence.seq))
                outfile.write("\n")

    def get_dataset(self):
        ungapped_records = []
        for record in self.msa:
            rec = SeqRecord(Seq(str(record.seq).replace(GAP_CHAR, "")), id=record.id)
            ungapped_records.append(rec)
        dataset = Dataset(ungapped_records, self.data_type, self._sorted)
        return dataset

    def get_shape(self):
        return
