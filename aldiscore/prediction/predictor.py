from Bio.SeqRecord import SeqRecord
from aldiscore.prediction.extractor import AlDIFeatureExtractor
import pandas as pd
import numpy as np
from typing import Literal
import joblib
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    from sklearn.pipeline import Pipeline


_ROOT = Path(__file__).parent
_MODEL_DIR = "models"


class AlDiPredictor:
    def __init__(self, model: Literal["latest", "vX.Y"] = "latest"):
        self.model: "Pipeline" = joblib.load(_ROOT / _MODEL_DIR / model)

    def predict(self, sequences: list[SeqRecord]):

        # extract features
        feat_df = AlDIFeatureExtractor(sequences).compute()

        # predict difficulty
        pred = self.model.predict(feat_df)

        return pred
