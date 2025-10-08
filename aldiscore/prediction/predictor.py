from aldiscore.prediction.extractor import FeatureExtractor
from typing import Literal
from pathlib import Path
import os
from typing import List, Union
from aldiscore import get_from_config, ROOT
from aldiscore.constants.constants import GAP_CHAR
import lightgbm as lgb
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class DifficultyPredictor:
    def __init__(
        self,
        model: Union["lgb.Booster", Literal["latest", "vX.Y"], Path] = "latest",
        seed: int = 0,
    ):
        if self._is_path(model):
            self.model: "lgb.Booster" = lgb.Booster(model_file=Path(model))
        elif isinstance(model, str):
            file_name = model
            if model == "latest":
                file_name = get_from_config("models", "latest")
            file_name += ".txt"
            model_path = ROOT / "models" / file_name
            self.model: "lgb.Booster" = lgb.Booster(model_file=model_path)
        else:  # Try to use directly
            self.model = model

        self.seed = seed

    def predict(
        self,
        sequences: Union[List[SeqRecord], List[str], Path],
        in_format: str = "fasta",
        drop_gaps: bool = True,
    ) -> float:
        # ensure correct input format
        if self._is_path(sequences):
            _sequences = list(SeqIO.parse(sequences, format=in_format))
        elif isinstance(sequences[0], str):
            _sequences = [SeqRecord(Seq(seq)) for seq in sequences]
        else:  # Assuming type SeqRecord here!
            _sequences = sequences

        if drop_gaps:
            records = []
            for seq in _sequences:
                gapless = SeqRecord(Seq(str(seq.seq).replace(GAP_CHAR, "")), id=seq.id)
                records.append(gapless)
            _sequences = records

        # extract features
        feat_df = FeatureExtractor(_sequences, seed=self.seed).compute()
        model_feats = self.model.feature_name()
        feat_df = feat_df[model_feats]
        # predict difficulty
        pred = self.model.predict(feat_df)[0]
        return pred

    def save(self, file_name: str) -> Path:
        models = os.listdir(ROOT / "models")
        if file_name in models:
            raise ValueError(f"File '{file_name}' exists already")
        else:
            self.model.save_model(ROOT / "models" / file_name)

        return ROOT / "models" / file_name

    def _is_path(self, input):
        return isinstance(input, Path) or (
            isinstance(input, str) and os.path.exists(input)
        )
