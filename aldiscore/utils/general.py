import datetime
import pandas as pd
import functools
import re
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from enums.enums import DataTypeEnum
import numpy as np


def compose_file_name(*args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    return "_".join([timestamp] + list(args))


def extract_matching_columns(df: pd.DataFrame, substring: str):
    """Extract columns from dataframe that contain the substring."""
    return df[df.columns[df.columns.str.contains(substring)]]


# # # # # # Sorting sequences # # # # #


def unique_representation(msa: MultipleSeqAlignment):
    # 1. order sequences with argsort_seq_order
    # 2. order columns with argsort_col_order
    pass


def argsort_col_order(msa: MultipleSeqAlignment):
    # TODO: before sequence ordering, enforce unique representation of ambiguous neighbor columns.
    # e.g.  ABABAD-B
    #       ABABA-A-
    #       ABABA--B
    # e.g.  ABABADB-
    #       ABABA--A
    #       ABABA-B-
    # 1. Identify regions of non-overlapping neighbors
    # 2. In each region:
    pass


def _compare_ids(idx_a, idx_b, msa):
    ords_a = [ord(char) for char in msa[idx_a].id]
    ords_b = [ord(char) for char in msa[idx_b].id]
    for ord_a, ord_b in zip(ords_a, ords_b):
        if ord_a != ord_b:
            return ord_a - ord_b
    if len(ords_a) != len(ords_b):
        return len(ords_a) - len(ords_b)
    else:
        return 0


def argsort_seq_order(msa: MultipleSeqAlignment, use_ids=True):
    if use_ids:
        sorting_func = _compare_ids
    else:
        sorting_func = _compare_ungapped_rows
    return sorted(
        range(len(msa)),
        key=functools.cmp_to_key(functools.partial(sorting_func, msa=msa)),
    )


DNA_CHARS = pd.Series(list("ACGT"))
PROTEIN_CHARS = pd.Series(list("ACDEFGHIKLMNPQRSTVWY"))


def infer_data_type(sequences: list[SeqRecord] | MultipleSeqAlignment):
    ungapped = None
    for seq in sequences:
        ungapped = str(seq.seq).replace("-", "")[:1000].upper()
        if len(ungapped) >= 100:
            break
    char_counts = pd.Series(list(ungapped)).value_counts()
    unique_chars = char_counts.index.to_series()

    dna_included = DNA_CHARS.isin(unique_chars)
    protein_included = PROTEIN_CHARS.isin(unique_chars)

    protein_missing = PROTEIN_CHARS[~protein_included]
    missing_counts = pd.Series(0, index=protein_missing)
    char_counts = pd.concat((char_counts, missing_counts))

    # If all DNA chars are included and their cumulative share is above 90%: DNA!
    if dna_included.all() & (char_counts[DNA_CHARS].sum() / len(ungapped) >= 0.9):
        return DataTypeEnum.DNA
    # If the cumulative share of protein chars is above 90%: Protein!
    elif char_counts[PROTEIN_CHARS].sum() / len(ungapped) >= 0.9:
        return DataTypeEnum.AA
    else:
        raise AssertionError(
            "Unknown alphabet detected for string '{}'".format(ungapped)
        )


def map_ungapped_to_gapped(values: list[np.ndarray], Q: list[np.ndarray], gap_value=0):
    K = len(Q)
    assert [len(values[k]) for k in range(K)] == [len(Q[k]) for k in range(K)]
    L = np.max([Q[k][-1] for k in range(K)]) + 1
    values_gapped = np.full((K, L), fill_value=gap_value)
    for k in range(K):
        values_gapped[k, Q[k]] = values[k]
    return values_gapped
