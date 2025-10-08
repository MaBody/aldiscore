import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
from Bio import AlignIO
from aldiscore import get_from_config
from aldiscore.datastructures.ensemble import Ensemble


def compute_pythia_difficulty(
    ensemble: Ensemble, raxmlng_path: Path = None, threads: int = 1
):
    try:
        import pypythia
        import pypythia.predictor
        import pypythia.prediction
        import pypythia.raxmlng
        import pypythia.custom_types
        import pypythia.custom_errors
        import pypythia.msa
    except ImportError:
        raise ImportError(
            "To use pythia, you need to install it first."
            "You can do so with: pip install pythiaphylopredictor"
        )

    PYTHIA = pypythia.predictor.DifficultyPredictor()
    if raxmlng_path is None:
        # Try to read from config files
        raxmlng_path = get_from_config("tools", "raxmlng")
        if raxmlng_path is None:
            raise ValueError("Specify raxmlng path in config or as a parameter!")
    RAXMLNG = pypythia.raxmlng.RAxMLNG(raxmlng_path)

    scores = []
    for alignment in ensemble.alignments:

        with NamedTemporaryFile() as tmpfile:
            file = Path(tmpfile.name)
            AlignIO.write([alignment.msa], file, "fasta")
            msa = pypythia.msa.parse_msa(file, pypythia.custom_types.FileFormat.FASTA)
            try:
                feats = pypythia.prediction.collect_features(
                    msa=msa, msa_file=file, raxmlng=RAXMLNG, threads=threads
                )
                scores.append(PYTHIA.predict(feats)[0])
            except pypythia.custom_errors.RAxMLNGError:
                scores.append(np.nan)
    return np.array(scores)
