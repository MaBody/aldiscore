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
    """
    Predicts alignment difficulty for a set of biological sequences.

    Uses a trained LightGBM model to predict how difficult it will be to correctly
    align a given set of sequences. The prediction is based on features extracted
    from the sequences using the FeatureExtractor class.

    The predictor can:
    1. Load pre-trained models from files or use custom models
    2. Process sequences in various input formats (FASTA files, strings, SeqRecords)
    3. Handle both DNA and protein sequences
    4. Optionally remove gaps from input sequences

    Attributes:
        _GROUP_SIZE (int): Number of sequences used in transitive consistency calculations
    """

    _GROUP_SIZE = 3  # Number of sequences per transitive consistency group

    def __init__(
        self,
        model: Union["lgb.Booster", Literal["latest", "vX.Y"], Path] = "latest",
        max_samples: int = 100,
        seed: int = 0,
    ):
        """
        Initialize the difficulty predictor.

        Args:
            model: The model to use for prediction. Can be:
                  - Path to a model file
                  - "latest" to use the most recent version
                  - A version string like "v1.0"
                  - A pre-loaded LightGBM model
            max_samples: Maximum number of sequence triplets to sample.
                       Controls computation time vs. prediction stability.
            seed: Random seed for reproducible sampling.

        Raises:
            ValueError: If the model file cannot be found or loaded
        """
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

        self._max_samples = max_samples
        self.seed = seed

    def predict(
        self,
        sequences: Union[str, Path, List[SeqRecord], List[str]],
        in_format: str = "fasta",
        in_type: Literal["DNA", "AA", "auto"] = "auto",
        drop_gaps: bool = True,
    ) -> float:
        """
        Predict alignment difficulty for a set of sequences.

        Args:
            sequences: Input sequences, either a path or BioPython objects.
            in_format: File format if sequences is a Path (e.g., "fasta")
            in_type: Type of sequences - "DNA", "AA" or "auto" for detection
            drop_gaps: Whether to remove gaps from input sequences. Convenient if input file is aligned.

        Returns:
            float: Predicted difficulty score
                  Higher values indicate sequences that are harder to align

        Raises:
            ValueError: If feature extraction fails
        """
        _sequences = None
        # ensure correct input format
        if isinstance(sequences, str) or isinstance(sequences, Path):
            _sequences = list(SeqIO.parse(sequences, format=in_format))
        elif isinstance(sequences, list) and (len(sequences) > 0):
            if isinstance(sequences[0], str):
                _sequences = [SeqRecord(Seq(seq)) for seq in sequences]
            elif isinstance(sequences[0], SeqRecord):
                _sequences = sequences

        if _sequences is None:
            raise ValueError(f"Detected wrong input format for parameter 'sequences'")

        if drop_gaps:
            records = []
            for seq in _sequences:
                gapless = SeqRecord(Seq(str(seq.seq).replace(GAP_CHAR, "")), id=seq.id)
                records.append(gapless)
            _sequences = records

        # Initialize PSA config dict with max number of triplets/pairs
        max_psa_count = self._max_samples * self._GROUP_SIZE
        self._psa_config = {"MAX_PSA_COUNT": max_psa_count}

        # extract features
        feat_df = FeatureExtractor(
            sequences=_sequences,
            psa_config=self._psa_config,
            track_perf=False,
            data_type=in_type,
            seed=self.seed,
        ).compute()

        model_feats = self.model.feature_name()
        feat_df = feat_df[model_feats]

        # predict difficulty
        pred = self.model.predict(feat_df)[0]

        return pred

    def save(self, file_name: str) -> Path:
        """
        Save the current model to a file.

        Args:
            file_name: Name of the file to save the model to
                      Will be saved in the package's models directory

        Returns:
            Path: Full path to the saved model file

        Raises:
            ValueError: If a file with the given name already exists
        """
        models = os.listdir(ROOT / "models")
        if file_name in models:
            raise ValueError(f"File '{file_name}' exists already")
        else:
            self.model.save_model(ROOT / "models" / file_name)

        return ROOT / "models" / file_name

    def _is_path(self, input) -> bool:
        """
        Check if the input is a valid file path.

        Args:
            input: Object to check

        Returns:
            bool: True if input is a Path object or an existing file path string
        """
        return isinstance(input, Path) or (
            isinstance(input, str) and os.path.exists(input)
        )
