import numpy as np
import pytest

from aldiscore.prediction.predictor import DifficultyPredictor


def test_predict_returns_score(demo_fasta):
    score = DifficultyPredictor(model="aa", seed=0).predict(demo_fasta, in_type="AA")
    assert isinstance(score, (float, np.floating))
    assert np.isfinite(score)


def test_predict_is_seed_reproducible(demo_fasta):
    first = DifficultyPredictor(model="aa", seed=0).predict(demo_fasta, in_type="AA")
    second = DifficultyPredictor(model="aa", seed=0).predict(demo_fasta, in_type="AA")
    assert first == second


def test_default_model_follows_in_type(demo_fasta, dna_seqs):
    explicit_aa = DifficultyPredictor(model="aa", seed=0).predict(
        demo_fasta, in_type="AA"
    )
    default_aa = DifficultyPredictor(seed=0).predict(demo_fasta, in_type="AA")
    assert default_aa == explicit_aa

    explicit_dna = DifficultyPredictor(model="dna", seed=0).predict(
        dna_seqs, in_type="DNA"
    )
    default_dna = DifficultyPredictor(seed=0).predict(dna_seqs, in_type="DNA")
    assert default_dna == explicit_dna
    assert default_dna != default_aa


def test_missing_in_type_is_rejected(demo_fasta):
    with pytest.raises(TypeError, match=r"in_type"):
        DifficultyPredictor(model="aa", seed=0).predict(demo_fasta)


def test_auto_in_type_is_rejected(demo_fasta):
    with pytest.raises(ValueError, match=r"Unknown data_type 'AUTO'"):
        DifficultyPredictor(model="aa", seed=0).predict(demo_fasta, in_type="auto")
