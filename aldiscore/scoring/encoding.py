import numpy as np
import pandas as pd
from aldiscore.enums.enums import PositionalEncodingEnum
from aldiscore.constants.constants import GAP_CHAR, GAP_CONST


def gapped_index_mapping(A: np.ndarray, dtype: np.dtype = np.int32):
    msa_idx = np.arange(A.shape[1], dtype=dtype)
    Q = []
    # Compute the indices of sites in the alignment
    for seq in A:
        Q_k = msa_idx[seq != GAP_CHAR]
        Q.append(Q_k)
    return Q


def encode_positions(
    A: np.ndarray,
    encoding: PositionalEncodingEnum,
    int_dtype: np.dtype = np.int32,
):
    """
    Encodes MSA as whole numbers (sites positive, gaps negative) for efficient homology set comparison.

    Parameters
    ----------
    A : np.ndarray
        Alignment in array form.
    L_max : int
        The maximum length of the ungapped sequences. Needed for scaling the sequence offset values.
    encoding : PositionalEncodingEnum
        - "uniform"  --> encode all gaps with "-1"
        - "sequence" --> encode all gaps in the same sequence identically
        - "position" --> encode each gap region in a sequence with the index of site to the left
        - "raw"      --> encode each char with its unicode number
        - "position_abs" --> encode each gap region in a sequence with its "gap-only" index (independent of the site indices)

    Example
    -------
    .. code-block:: python

      [["a", "b", "a", "-", "-", "c"],
       ["-", "b", "-", "b", "c", "b"]]
      # uniform
      [[ 11,  12,  13,  -1,  -1,  14],
       [ -1,  21,  -1,  22,  23,  24]]
      # sequence
      [[ 11,  12,  13, -11, -11,  14],
       [-21,  21, -21,  22,  23,  24]]
      # position
      [[ 11,  12,  13, -13, -13,  14],
       [-20,  21, -21,  22,  23,  24]]
      # raw
      [[ 97,  98,  97,  -1,  -1,  99],
       [ -1,  98,  -1,  98,  99,  98]]
      # gap_regions
      [[ 0,    0,   0,   2,   2,   0],
       [ 1,    0,   1,   0,   0,   0]]

    """
    site_codes = np.zeros(A.shape, dtype=int_dtype)
    gap_mask = A == GAP_CHAR

    L_max = (~gap_mask).sum(axis=1).max()
    # The number of non-consecutive gap regions in a sequence is bounded by L_max + 1
    # --> The smallest L_max + 1 positions are reserved for the position in the sequence
    # --> All larger positions are used for the sequence index k
    offset_magnitude = 10 ** (np.ceil(np.log10(L_max + 2)))
    seq_offsets = (offset_magnitude * np.arange(1, len(A) + 1)).astype(int_dtype)
    seq_offsets = seq_offsets[:, np.newaxis]
    site_encoding = (~gap_mask).cumsum(axis=1, dtype=int_dtype) + seq_offsets

    match encoding:
        case PositionalEncodingEnum.UNIFORM:
            gap_encoding = -gap_mask.astype(int_dtype)

        case PositionalEncodingEnum.SEQUENCE:
            gap_encoding = -gap_mask.astype(int_dtype) - seq_offsets

        case PositionalEncodingEnum.POSITION:
            gap_encoding = -(~gap_mask).cumsum(axis=1, dtype=int_dtype) - seq_offsets

        case PositionalEncodingEnum.RAW:
            code_func = np.vectorize(ord)
            site_codes = code_func(A)
            site_codes[A == GAP_CHAR] = GAP_CONST
            site_codes = site_codes.astype(int_dtype)
            return site_codes

        case PositionalEncodingEnum.GAP_REGIONS:
            # gap_ends contains True if after a gap there is a normal site (i.e., the ends of gapped regions)
            gap_ends = np.diff(gap_mask, n=1, axis=1, append=0) == -1
            # gap_counts contains an incrementing counter for gapped positions in each sequence
            gap_counts = gap_mask.cumsum(axis=1)
            # Set all positions to zero that are not the end of a gap region
            gap_counts[~gap_ends] = 0
            # For each gap region, the last gap now holds the count of all gaps up to that point
            # Right-shift the gap end counters (counts[i] = counts_shifted[i+1])
            gap_end_counts_shifted = np.roll(gap_counts, shift=1, axis=1)
            # Set the first position to zero (no preceding gap regions)
            gap_end_counts_shifted[:, 0] = 0
            # Forward fill the values: now position i holds the cumulative gap count prior to the current gap region
            gap_end_counts_shifted = _ffill_numpy_2d_axis_1(
                gap_end_counts_shifted, val=0
            )
            # Compute the length of the gap regions at their last position
            site_codes = gap_counts - gap_end_counts_shifted
            # Remove all positions that are not gap region ends
            site_codes[~gap_ends] = 0
            # Backward fill with gap region end values
            site_codes = _ffill_numpy_2d_axis_1(site_codes[:, ::-1], val=0)[:, ::-1]
            # Remove values at non-gap sites
            site_codes[~gap_mask] = 0
            # At each gap, A_code now contains the length of the current gap region at every gap position
            site_codes = site_codes.astype(int_dtype)
            return site_codes

        case _:
            raise ValueError(encoding)

    site_codes = np.where(gap_mask, gap_encoding, site_encoding).astype(int_dtype)
    return site_codes


def _ffill_numpy_2d_axis_1(A: np.ndarray, val: int | float):
    axis = 1
    """Forward fill A where A == val for 2d array along axis 1."""
    mask = A == val
    idx = np.where(~mask, np.arange(mask.shape[axis]), 0)
    idx = np.maximum.accumulate(idx, axis=axis)
    out = A[np.arange(idx.shape[0])[:, np.newaxis], idx]
    return out


def _ffill_numpy_3d_axis_2(A: np.ndarray, val: int | float):
    """Forward fill A where A == val for 3d array along axis 2."""
    axis = 2
    mask = A == val
    idx = np.where(~mask, np.arange(mask.shape[axis]), 0)
    idx = np.maximum.accumulate(idx, axis=axis)
    out = A[
        np.arange(idx.shape[0])[:, np.newaxis, np.newaxis],
        np.arange(idx.shape[1])[np.newaxis, :, np.newaxis],
        idx,
    ]
    return out
