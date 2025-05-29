import pathlib
import subprocess
import tempfile
from collections import defaultdict
from typing import Optional
import itertools
import numpy as np
from numpy import typing as npt
from scipy.fft import dct
import pandas as pd
from enums.enums import DataTypeEnum

DNA_CHARS = "N-ACGT"  # Include N for safety
DNA_CHAR_MAP = defaultdict(lambda: 0, {"A": 63, "C": 127, "G": 191, "T": 255})
AA_CHARS = [char for char in "-ACDEFGHIKLMNOPQRSTUVWY"]
# Creating 23 numbers from 0 to 255 with maximal spacing
AA_MAPPING = np.arange(len(AA_CHARS)) / (len(AA_CHARS) - 1) * 255
# Round to intergers (spacing of 11 or 12)
AA_MAPPING = np.round(AA_MAPPING, decimals=0).astype(int)
AA_CHAR_MAP = defaultdict(
    lambda: 0, {char: mapping for char, mapping in zip(AA_CHARS, AA_MAPPING)}
)


def run_cmd(
    cmd: list[str],
    outfile: pathlib.Path,
    logfile: Optional[pathlib.Path] = None,
    log_write_mode: str = "w",
    out_write_mode: str = "w",
) -> None:
    """
    Runs a given command using the subprocess Python module. The results of the run are stored in the given `outfile`
    and all logging and errors are written to `logfile` using the given write mode.

    Parameters
    ----------
    cmd : list[str]
        Command to run as list of strings.
    outfile : pathlib.Path
        File where to store the output of the given command execution in.
    logfile : pathlib.Path
        File where to store all logging/errors in.
    log_write_mode : str
        Write mode to open the logfile with.
        Allowed options are `'a'` (append logs to existing logs) and `'w'` (overwrite all existing logs).

    """
    if log_write_mode not in ["a", "w"]:
        raise ValueError(
            f"Invalid write mode for logfile given: {log_write_mode}. Allowed options are 'a' (append) and 'w' (write)."
        )
    if not logfile:
        logfile = pathlib.Path(tempfile.NamedTemporaryFile("w").name)
    with logfile.open(log_write_mode) as log, outfile.open(out_write_mode) as out:
        try:
            subprocess.run(cmd, stdout=out, stderr=log, check=True, encoding="utf-8")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Error running CMD: {' '.join(cmd)}"
                f"Check the logfile {logfile.absolute()} for details."
                f"Logfile: {logfile.open().read()}"
            )


def _perceptual_hash(image: npt.NDArray, size: Optional[int]) -> str:
    dct_coeffs = dct(dct(image, axis=0), axis=1)
    binary_sequence = np.clip(np.sign(dct_coeffs), 0, 1)

    if size:
        binary_sequence = binary_sequence[:size, :size]

    hash_value = "".join([str(int(bit)) for bit in binary_sequence.flatten()])
    return hash_value


def sequence_perceptual_hash(
    sequence: npt.NDArray, data_type: DataTypeEnum, size: Optional[int] = None
) -> str:
    """
    Computes the perceptual hash for a given DNA sequence.
    Perceptual hashing originates in computer vision to computes the similarity between two images.
    To compute the perceptual hash for a single DNA sequence, we first have to transform the char sequence into an
    image-like matrix, i.e. a 2D numpy array.

    The perceptual hash is computed using the following steps:
    1. Encoding: The DNA sequence is encoded as numeric vector on a scale of 0 to 255 (like a grey-scale image).
        Gaps or fully undetermined characters (`-` and `N`) correspond to 0, `T` corresponds to 255 and `A`, `C`, `G`
        are assigned values such that their distance is maximal (63, 127, 191, see `DNA_CHAR_MAPPING` above).
    2. This numeric vector is reshaped into a quadratic 2D vector using the square root of the sequence length.
    3. Next, we compute the Discrete Cosine Transform (DCT) along the rows and subsequently along the columns.
    4. The resulting sequence is transformed into a bit sequence using the sign of the DCT coefficients and clipping between 0 and 1.
    5. The hash value is then the string of the first `size` bits of this bit sequence.

    Parameters
    ----------
    sequence : npt.NDArray
        DNA sequence to compute the perceptual hash for. The sequence is expected to be a char array.
    size : Optional[int]
         Length of the hash to compute and compare. Default is to compute and return the full length.

    Returns
    -------
    str
        The perceptual hash of the given DNA sequence.

    """
    side_length = int(np.ceil(np.sqrt(len(sequence))))
    if data_type == DataTypeEnum.DNA:
        numeric_seq = pd.Series(sequence).map(lambda x: DNA_CHAR_MAP[x]).to_numpy()
    elif data_type == DataTypeEnum.AA:
        numeric_seq = pd.Series(sequence).map(lambda x: AA_CHAR_MAP[x]).to_numpy()
    else:
        raise ValueError(f"Data type not recognized: {data_type}")
    numeric_seq_2d = np.resize(numeric_seq, (side_length, side_length))

    return _perceptual_hash(numeric_seq_2d, size)


def msa_perceptual_hash(
    msa: npt.NDArray, data_type: DataTypeEnum, size: Optional[int] = None
) -> str:
    """
    Computes the perceptual hash for a given DNA MSA.
    Perceptual hashing originates in computer vision to computes the similarity between two images.

    The perceptual hash is computed using the following steps:
    1. Encoding: The DNA MSA is encoded as numeric 2D matrix on a scale of 0 to 255 (like a grey-scale image).
        Gaps or fully undetermined characters (`-` and `N`) correspond to 0, `T` corresponds to 255 and `A`, `C`, `G`
        are assigned values such that their distance is maximal (63, 127, 191, see `DNA_CHAR_MAPPING` above).
    2. Next, we compute the Discrete Cosine Transform (DCT) along the rows and subsequently along the columns.
    3. The resulting sequence is transformed into a bit sequence using the sign of the DCT coefficients and clipping between 0 and 1.
    4. The hash value is then the string of the first `size` bits of this bit sequence.

    Parameters
    ----------
    msa : npt.NDArray
        DNA MSA to compute the perceptual hash for. The MSA is expected to be a 2D char array.
    size : Optional[int]
         Length of the hash to compute and compare. Default is to compute and return the full length.

    Returns
    -------
    str
        The perceptual hash of the given DNA MSA.

    """
    if data_type == DataTypeEnum.DNA:
        numeric_msa = pd.DataFrame(msa).map(lambda x: DNA_CHAR_MAP[x]).to_numpy()
    elif data_type == DataTypeEnum.AA:
        numeric_msa = pd.DataFrame(msa).map(lambda x: AA_CHAR_MAP[x]).to_numpy()
    else:
        raise ValueError(f"Data type not recognized: {data_type}")
    return _perceptual_hash(numeric_msa, size)


def absolute_hamming_dist(hash1: str, hash2: str) -> int:
    """
    Computes the absolute hamming distance between the given strings.

    Parameters
    ----------
    hash1 : str
        The first string for the pairwise hamming distance computation.
    hash2 : str
        The second string for the pairwise hamming distance computation.

    Returns
    -------
    int
        The hamming distance between the given strings.

    """
    return sum([1 if h1 != h2 else 0 for h1, h2 in zip(hash1, hash2)])


def normalized_hamming_dist(hash1: str, hash2: str) -> float:
    """
    Computes the normalized hamming distance between the given strings.
    The normalized hamming distance corresponds to the absolute hamming distance divided by the length of the first string.

    Parameters
    ----------
    hash1 : str
        The first string for the pairwise hamming distance computation.
    hash2 : str
        The second string for the pairwise hamming distance computation.

    Returns
    -------
    int
        The normalized hamming distance between the given strings.
    """
    return absolute_hamming_dist(hash1, hash2) / len(hash1)


def format_dist_mat(dists: list, ensemble):
    """
    Reconstructs the distance matrix based on a list of pairwise distances.

    Parameters
    ----------
    - dists: list
        Flat list of computed distances for the given ensemble.
    - ensemble: Ensemble
        Ensemble of alignments on the same dataset.

    Returns
    -------
    - dist_mat: pd.DataFrame
        Symmetric distance matrix.
    """
    n_alignments = len(ensemble.ensemble)
    dist_mat = np.zeros((n_alignments, n_alignments), dtype=np.float32)
    # Puzzling together the distance matrix
    tool_names = [""] * n_alignments
    mat_idxs = range(n_alignments)
    for list_idx, (idx_x, idx_y) in enumerate(itertools.combinations(mat_idxs, r=2)):
        # Convert the flat list of distances back to a distance matrix
        # This works because of the stability of itertools.combinations
        dist = dists[list_idx]
        dist_mat[idx_x, idx_y] = dist
        # Store the name of the aligner that produced the MSA
        tool_names[idx_x] = ensemble.ensemble[idx_x].tool
        tool_names[idx_y] = ensemble.ensemble[idx_y].tool
    # Copy symmetric values, diagonal is always zero
    dist_mat = dist_mat + dist_mat.T
    # Convert to df, name rows/cols with associated aligner
    col_names = [name + "_" + str(idx) for name, idx in zip(tool_names, mat_idxs)]
    dist_mat = pd.DataFrame(dist_mat, index=tool_names, columns=col_names)
    return dist_mat


def compute_custom_metrics(idxs: tuple, ensemble=None, metrics=None):
    idx_x, idx_y = idxs
    dist_dict = {}
    for name, metric in metrics.items():
        dist = metric.compute(
            ensemble.ensemble[idx_x],
            ensemble.ensemble[idx_y],
        )
        # print(f"{name} : {time()- start}s")
        dist_dict[name] = dist
    return dist_dict


DATA_TYPE_MAP = {DataTypeEnum.DNA: DataType.DNA, DataTypeEnum.AA: DataType.AA}
DATA_TYPE_MAP.update(dict(zip(DATA_TYPE_MAP.values(), DATA_TYPE_MAP.keys())))


def convert_data_type_enum(data_type: DataType | DataTypeEnum):
    if data_type in DATA_TYPE_MAP:
        return DATA_TYPE_MAP[data_type]
    else:
        raise ValueError(f"Did not recognize data_type = {data_type}")
