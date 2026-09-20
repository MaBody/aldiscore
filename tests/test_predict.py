import numpy as np

from aldiscore.prediction.predictor import DifficultyPredictor


def test_predict_returns_score(demo_fasta):
    score = DifficultyPredictor(model="aa", seed=0).predict(demo_fasta, in_type="AA")
    assert isinstance(score, (float, np.floating))
    assert np.isfinite(score)


def test_predict_is_seed_reproducible(demo_fasta):
    first = DifficultyPredictor(model="aa", seed=0).predict(demo_fasta, in_type="AA")
    second = DifficultyPredictor(model="aa", seed=0).predict(demo_fasta, in_type="AA")
    assert first == second


def test_auto_type_detection_matches_explicit(demo_fasta):
    explicit = DifficultyPredictor(model="aa", seed=0).predict(demo_fasta, in_type="AA")
    auto = DifficultyPredictor(model="aa", seed=0).predict(demo_fasta, in_type="auto")
    assert explicit == auto
