from aldiscore.prediction.extractor import FeatureExtractor
import pandas as pd
import numpy as np
from typing import Literal
import joblib  # TODO: switch to safer way of model loading
from pathlib import Path
import os
from typing import TYPE_CHECKING
from aldiscore import get_from_config, ROOT, MODEL_DIR
from aldiscore.constants.constants import GAP_CHAR
import lightgbm as lgb
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class DifficultyPredictor:
    def __init__(
        self, model: "lgb.Booster" | Literal["latest", "vX.Y"] | Path = "latest"
    ):
        if self._is_path(model):
            self.model: "lgb.Booster" = lgb.Booster(model_file=Path(model))
        elif isinstance(model, str):
            file_name = model
            if model == "latest":
                file_name = get_from_config("models", "latest")
            file_name += ".txt"
            model_path = MODEL_DIR / file_name
            self.model: "lgb.Booster" = lgb.Booster(model_file=model_path)
        else:  # Try to use directly
            self.model = model

    def predict(
        self,
        sequences: list[SeqRecord | str] | Path,
        in_format: str = "fasta",
        drop_gaps=False,
    ) -> float:

        # ensure correct input format
        if self._is_path(sequences):
            _sequences = list(SeqIO.parse(sequences, format=in_format))
        elif isinstance(_sequences[0], str):
            _sequences = [SeqRecord(Seq(seq)) for seq in sequences]
        else:
            _sequences = sequences

        if drop_gaps:
            records = []
            for seq in _sequences:
                gapless = SeqRecord(Seq(str(seq.seq).replace(GAP_CHAR, "")), id=seq.id)
                records.append(gapless)
            _sequences = records

        # extract features
        feat_df = FeatureExtractor(_sequences).compute()
        model_feats = self.model.feature_name()
        feat_df = feat_df[model_feats]
        # predict difficulty
        pred = self.model.predict(feat_df)[0]
        return pred

    def save(self, file_name: str) -> Path:
        models = os.listdir(MODEL_DIR)
        if file_name in models:
            raise ValueError(f"File '{file_name}' exists already")
        else:
            self.model.save_model(MODEL_DIR / file_name)

        return MODEL_DIR / file_name

    def _is_path(self, input):
        return isinstance(input, Path) or (
            isinstance(input, str) and os.path.exists(input)
        )
