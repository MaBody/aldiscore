import pathlib
from Bio import SeqIO
from datastructures.alignment import Alignment
from datastructures.dataset import Dataset
import datastructures.utils
from enums.enums import DataTypeEnum


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

    # @classmethod
    # def from_efa(
    #     cls,
    #     muscle5_path: pathlib.Path,
    #     efa_file: pathlib.Path,
    #     data_type: DataTypeEnum = None,
    #     dataset: Dataset = None,
    # ):
    #     ensemble = [
    #         Alignment(str(file), data_type=data_type)
    #         for file in explode_efa(muscle5_path, efa_file)
    #     ]
    #     return Ensemble(ensemble, dataset=dataset)

    # def to_efa(self, efa_file: pathlib.Path) -> None:
    #     """
    #     Stores the ensemble of alignments to file using the Muscle5 `.efa` file format.

    #     Parameters
    #     ----------
    #     efa_file : pathlib.Path
    #         File where to write the ensemble to.

    #     """
    #     with efa_file.open("w") as f:
    #         for i, alignment in enumerate(self.alignments):
    #             f.write(f"<msa.{i}\n")
    #             SeqIO.write(alignment.msa, f, "fasta")
    #             f.write("\n")


# def explode_efa(muscle5: pathlib.Path, efa_file: pathlib.Path) -> list[pathlib.Path]:
#     """
#     Stores all alignments in the given `efa_file` as separate alignment files in FASTA format and returns their file paths.

#     Parameters
#     ----------
#     muscle5 : pathlib.Path
#         Path to the Muscle5 executable. The Muscle5 `-efa_explode` command is used to separate the alignments.
#     efa_file : pathlib.Path
#         Path to the `.efa` file to extract the alignments for.

#     Returns
#     -------
#     list[pathlib.Path]:
#         List of file paths for the separate alignments in FASTA format.

#     """
#     cmd = [
#         str(muscle5),
#         "-efa_explode",
#         str(efa_file),
#         "-prefix",
#         str(efa_file.parent) + "/",
#         "-suffix",
#         ".msa.fasta",
#     ]
#     logfile = efa_file.parent / "efa_explode.log"
#     run_cmd(cmd, outfile=logfile, logfile=logfile)

#     return [
#         file for file in efa_file.parent.iterdir() if file.name.endswith(".msa.fasta")
#     ]
