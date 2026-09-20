import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from aldiscore.prediction.predictor import DifficultyPredictor


@pytest.fixture(scope="module")
def protein_seqs(demo_fasta):
    return [str(r.seq).replace("-", "") for r in SeqIO.parse(demo_fasta, "fasta")]


@pytest.fixture(scope="module")
def protein_score(protein_seqs):
    return DifficultyPredictor(model="aa", seed=0).predict(protein_seqs, in_type="AA")


def _insert(seqs, char, at=5):
    return [s[:at] + char + s[at:] for s in seqs]


def test_stop_codon_is_rejected_with_sequence_id(protein_seqs):
    records = [SeqRecord(Seq(s), id=f"seq{i}") for i, s in enumerate(protein_seqs)]
    records[1] = SeqRecord(Seq(protein_seqs[1][:5] + "*" + protein_seqs[1][5:]), id="seq1")
    with pytest.raises(ValueError, match=r"Invalid characters.*\*.*seq1"):
        DifficultyPredictor(model="aa", seed=0).predict(records, in_type="AA")


def test_unknown_character_is_rejected(protein_seqs):
    with pytest.raises(ValueError, match=r"Invalid characters"):
        DifficultyPredictor(model="aa", seed=0).predict(_insert(protein_seqs, "?"), in_type="AA")


def test_dot_is_treated_as_gap(protein_seqs, protein_score):
    score = DifficultyPredictor(model="aa", seed=0).predict(_insert(protein_seqs, "..."), in_type="AA")
    assert score == protein_score


def test_lowercase_matches_uppercase(protein_seqs, protein_score):
    score = DifficultyPredictor(model="aa", seed=0).predict([s.lower() for s in protein_seqs], in_type="AA")
    assert score == protein_score


def test_protein_ambiguity_codes_are_accepted(protein_seqs):
    score = DifficultyPredictor(model="aa", seed=0).predict(_insert(protein_seqs, "XBZ"), in_type="AA")
    assert isinstance(float(score), float)


DNA_SEQS = [
    "ATGCGTACGTTAGCATCGATCGATCGTAGCTAGCTAGCTAGGCTAGCTAGCTAGCATCG",
    "ATGCGTACGTTAGCATCGATCGATCGTAGCTAGCTAGCTAGGCTAGCTAGCTAGCTTCG",
    "ATGCGTACGTTAGCATCGATCGATCGTAGCTAGCTTGCTAGGCTAGCTAGCTAGCATCG",
    "ATGCGTACGTTAGCATCGATCGATCGTAGCAAGCTAGCTAGGCTAGCTAGCTAGCATCG",
]


def test_dna_ambiguity_codes_are_accepted():
    score = DifficultyPredictor(model="dna", seed=0).predict(_insert(DNA_SEQS, "NR"), in_type="DNA")
    assert isinstance(float(score), float)


def test_dna_stop_codon_is_rejected():
    with pytest.raises(ValueError, match=r"Invalid characters for DNA"):
        DifficultyPredictor(model="dna", seed=0).predict(_insert(DNA_SEQS, "*"), in_type="DNA")
