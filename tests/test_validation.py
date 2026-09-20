import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from aldiscore.prediction.extractor import FeatureExtractor
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


def _records(seqs):
    return [SeqRecord(Seq(s), id=f"seq{i}") for i, s in enumerate(seqs)]


def test_empty_input_reports_sequence_count():
    with pytest.raises(ValueError, match=r"Need at least 3 sequences, found 0"):
        FeatureExtractor([])


def test_too_few_sequences_reports_count():
    with pytest.raises(ValueError, match=r"Need at least 3 sequences, found 2"):
        FeatureExtractor(_records(DNA_SEQS[:2]))


def test_too_few_sequences_only_warns_in_warn_mode(capsys):
    FeatureExtractor(_records(DNA_SEQS[:2]), validate="warn")
    assert "WARNING: Need at least 3 sequences" in capsys.readouterr().out


def test_lowercase_in_type_is_accepted():
    upper = DifficultyPredictor(model="dna", seed=0).predict(DNA_SEQS, in_type="DNA")
    lower = DifficultyPredictor(model="dna", seed=0).predict(DNA_SEQS, in_type="dna")
    assert lower == upper


def test_unknown_in_type_is_rejected():
    with pytest.raises(ValueError, match=r"Unknown data_type 'RNA'"):
        DifficultyPredictor(model="dna", seed=0).predict(DNA_SEQS, in_type="RNA")


def test_unknown_in_type_is_rejected_even_in_warn_mode():
    with pytest.raises(ValueError, match=r"Unknown data_type"):
        FeatureExtractor(_records(DNA_SEQS), data_type="RNA", validate="warn")
