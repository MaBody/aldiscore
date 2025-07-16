import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
from Bio import AlignIO

from aldiscore.datastructures.ensemble import Ensemble


# def _parse_pythia_ensemble(ensemble: Ensemble) -> list[MSA]:
#     msas = []
#     for alignment in ensemble.alignments:
#         msas.append[_parse_pythia_msa(alignment)]


# def _parse_pythia_msa(alignment: Alignment) -> list[pypythia.msa.MSA]:


def compute_pythia_difficulty(ensemble: Ensemble, raxml_path: Path, threads: int = 1):
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
    RAXMLNG = pypythia.raxmlng.RAxMLNG(raxml_path)

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
