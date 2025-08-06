from Bio.SeqRecord import SeqRecord
from aldiscore.prediction.extractor import FeatureExtractor
import pandas as pd
import numpy as np
from typing import Literal
import joblib  # TODO: switch to safer way of model loading
from pathlib import Path
import yaml
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    from sklearn.pipeline import Pipeline


_ROOT = Path(__file__).parent
_MODEL_DIR = "models"


class AlDiPredictor:
    def __init__(self, model: Literal["latest", "vX.Y"] | Path = "latest"):
        if isinstance(model, Path):
            model_path = model
        else:
            model_path = _ROOT / _MODEL_DIR
            file_name = model
            if model == "latest":
                file_name = yaml.load(open(_ROOT / "config.yaml"))["latest"]
            model_path = model_path / file_name
        self.model: "Pipeline" = joblib.load(model_path)

    def predict(self, sequences: list[SeqRecord]):
        # extract features
        feat_df = FeatureExtractor(sequences).compute()
        # predict difficulty
        pred = self.model.predict(feat_df)
        return pred

    def store(self, model: "Pipeline", file_name: str):
        models = os.listdir(_ROOT / _MODEL_DIR)
        if file_name in models:
            raise ValueError(f"File '{file_name}' exists already")
        else:
            joblib.dump(model, _ROOT / _MODEL_DIR / file_name)
