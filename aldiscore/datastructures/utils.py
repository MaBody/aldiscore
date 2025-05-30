import functools
import pandas as pd
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from aldiscore.enums.enums import DataTypeEnum
from aldiscore.constants.constants import DNA_CHARS, AA_CHARS, GAP_CHAR


def _compare_ungapped_rows(idx_a, idx_b, records):
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
    ords_a = [ord(char) for char in records[idx_a].id]
    ords_b = [ord(char) for char in records[idx_b].id]
    for ord_a, ord_b in zip(ords_a, ords_b):
        if ord_a != ord_b:
            return ord_a - ord_b
    if len(ords_a) != len(ords_b):
        return len(ords_a) - len(ords_b)
    else:
        return 0


def argsort_seq_order(records: list[SeqRecord] | MultipleSeqAlignment):
    use_ids = len(set([record.id for record in records])) == len(records)
    if use_ids:
        sorting_func = _compare_ids
    else:
        sorting_func = _compare_ungapped_rows
    return sorted(
        range(len(records)),
        key=functools.cmp_to_key(functools.partial(sorting_func, records=records)),
    )


# TODO: Remove if not needed
# def _sort_sequences(self, sequences: list):
#     # Get the maximum sequence length to determine the padding sizes
#     L_max = max([len(seq.seq) for seq in sequences])
#     # Add gaps as padding to simulate alignment (gaps are ignored in the sorting)
#     padded_sequences = []
#     for seq in sequences:
#         padded_seq = str(seq.seq) + "-" * (L_max - len(seq))
#         padded_sequences.append(SeqRecord(Seq(padded_seq), id=seq.id))
#     # padded_sequences = np.array(
#     #     [np.concatenate((seq, ["-"] * (L_max - len(seq)))) for seq in sequences]
#     # )
#     # Get the indices that sort the sequences (implementation only works for alignments)
#     sort_idxs = argsort_seq_order(MultipleSeqAlignment(padded_sequences))
#     return sort_idxs


def infer_data_type(records: list[SeqRecord] | MultipleSeqAlignment):
    ungapped = None
    for seq in records:
        ungapped = str(seq.seq).replace(GAP_CHAR, "")[:1000].upper()
        if len(ungapped) >= 100:
            break
    char_counts = pd.Series(list(ungapped)).value_counts()
    unique_chars = char_counts.index.to_series()

    dna_included = DNA_CHARS.isin(unique_chars)
    protein_included = AA_CHARS.isin(unique_chars)

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


def get_unique_key(records: list[SeqRecord] | MultipleSeqAlignment):
    ids = set([record.id for record in records])
    if len(set(ids)) == len(records):
        ids_immutable = tuple(sorted(list(ids)))
        key = hash(ids_immutable)
    else:
        key_str = "#".join(str(map(lambda record: str(record.seq), records)))
        key = hash(key_str)
    return key
