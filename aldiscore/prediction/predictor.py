from Bio.SeqRecord import SeqRecord
from aldiscore.prediction.extractor import FeatureExtractor
import pandas as pd
import numpy as np
from typing import Literal
import joblib  # TODO: switch to safer way of model loading
from pathlib import Path
import os
from typing import TYPE_CHECKING
from aldiscore import get_from_config, ROOT

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    from sklearn.pipeline import Pipeline


class DifficultyPredictor:
    _MODEL_DIR = ROOT / "prediction" / "models"

    def __init__(self, model: Pipeline | Literal["latest", "vX.Y"] | Path = "latest"):
        if isinstance(model, Path):
            model_path = model
            self.model: "Pipeline" = joblib.load(model_path)
        elif isinstance(model, str):
            file_name = model
            if model == "latest":
                file_name = get_from_config("models", model)
            model_path = self._MODEL_DIR / file_name
            self.model: "Pipeline" = joblib.load(model_path)
        else:  # Try to use directly
            self.model = model

    def predict(self, sequences: list[SeqRecord]):
        # extract features
        feat_df = FeatureExtractor(sequences).compute()
        # predict difficulty
        pred = self.model.predict(feat_df)
        return pred

    def store(self, model: "Pipeline", file_name: str):
        models = os.listdir(self._MODEL_DIR)
        if file_name in models:
            raise ValueError(f"File '{file_name}' exists already")
        else:
            joblib.dump(model, self._MODEL_DIR / file_name)
