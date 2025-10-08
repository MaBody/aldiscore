"""
Utility functions for sequence and alignment data structures.

Provides helper functions for sorting, comparing, and inferring types of biological sequence records,
as well as generating unique keys for collections of sequences.
"""

import functools
import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from aldiscore.enums.enums import DataTypeEnum
from aldiscore.constants.constants import DNA_CHARS, AA_CHARS, GAP_CHAR
from typing import List, Union


def _compare_ungapped_rows(idx_a, idx_b, records):
    """
    Compare two sequence records by their ungapped sequence content.

    Parameters
    ----------
    idx_a : int
        Index of the first sequence record.
    idx_b : int
        Index of the second sequence record.
    records : list[SeqRecord] | MultipleSeqAlignment
        Collection of sequence records.

    Returns
    -------
    int
        Negative if record at idx_a < idx_b, positive if >, zero if equal.
    """
    i, j = 0, 0
    len_record_a = len(records[idx_a])
    len_record_b = len(records[idx_b])
    min_len = min(len_record_a, len_record_b)
    while (i < min_len) and (j < min_len):
        if records[idx_a][i] == GAP_CHAR:
            i += 1
        elif records[idx_b][j] == GAP_CHAR:
            j += 1
        elif records[idx_a][i] == records[idx_b][j]:
            i += 1
            j += 1
        else:
            return ord(records[idx_a][i]) - ord(records[idx_b][j])
    return len_record_a - len_record_b


def _compare_ids(idx_a, idx_b, records):
    """
    Compare two sequence records by their IDs.

    Parameters
    ----------
    idx_a : int
        Index of the first sequence record.
    idx_b : int
        Index of the second sequence record.
    records : list[SeqRecord] | MultipleSeqAlignment
        Collection of sequence records.

    Returns
    -------
    int
        Negative if record at idx_a < idx_b, positive if >, zero if equal.
    """
    ords_a = [ord(char) for char in records[idx_a].id]
    ords_b = [ord(char) for char in records[idx_b].id]
    for ord_a, ord_b in zip(ords_a, ords_b):
        if ord_a != ord_b:
            return ord_a - ord_b
    if len(ords_a) != len(ords_b):
        return len(ords_a) - len(ords_b)
    else:
        return 0


def argsort_seq_order(records: Union[List[SeqRecord], MultipleSeqAlignment]):
    """
    Return the indices that would sort the records in a stable, canonical order.

    Uses IDs if all are unique, otherwise sorts by ungapped sequence content.

    Parameters
    ----------
    records : list[SeqRecord] | MultipleSeqAlignment
        Collection of sequence records.

    Returns
    -------
    list[int]
        Indices that sort the records.
    """
    use_ids = len(set([record.id for record in records])) == len(records)
    if use_ids:
        sorting_func = _compare_ids
    else:
        sorting_func = _compare_ungapped_rows
    return sorted(
        range(len(records)),
        key=functools.cmp_to_key(functools.partial(sorting_func, records=records)),
    )


def infer_data_type(records: Union[List[SeqRecord], MultipleSeqAlignment]):
    """
    Infer the biological data type (DNA or protein) from a collection of sequence records.

    Examines the first sufficiently long ungapped sequence and determines the type
    based on character composition.

    Parameters
    ----------
    records : list[SeqRecord] | MultipleSeqAlignment
        Collection of sequence records.

    Returns
    -------
    DataTypeEnum
        Inferred data type (DNA or AA).

    Raises
    ------
    AssertionError
        If the alphabet cannot be determined.
    """
    ungapped = None
    for seq in records:
        ungapped = str(seq.seq).replace(GAP_CHAR, "")[:1000].upper()
        if len(ungapped) >= 100:
            break
    char_counts = pd.Series(list(ungapped)).value_counts()
    unique_chars = char_counts.index.to_series()

    dna_included = np.isin(DNA_CHARS, unique_chars)
    protein_included = np.isin(AA_CHARS, unique_chars)

    protein_missing = AA_CHARS[~protein_included]
    missing_counts = pd.Series(0, index=protein_missing)
    char_counts = pd.concat((char_counts, missing_counts))

    # If all DNA chars are included and their cumulative share is above 90%: DNA!
    if dna_included.all() & (char_counts[DNA_CHARS].sum() / len(ungapped) >= 0.9):
        return DataTypeEnum.DNA
    # If the cumulative share of protein chars is above 90%: Protein!
    elif char_counts[AA_CHARS].sum() / len(ungapped) >= 0.9:
        return DataTypeEnum.AA
    else:
        raise AssertionError(
            "Unknown alphabet detected for string '{}'".format(ungapped)
        )


def get_unique_key(records: Union[List[SeqRecord], MultipleSeqAlignment]):
    """
    Generate a unique hash key for a collection of sequence records based on their sequence content.

    Parameters
    ----------
    records : list[SeqRecord] | MultipleSeqAlignment
        Collection of sequence records.

    Returns
    -------
    int
        Hash value representing the collection.
    """
    key_str = "#".join(map(lambda record: str(record.seq), records))
    key = hash(key_str)
    return key
