import argparse
import os
import sys

# import pandas as pd

# from aldiscore.prediction.extractor import FeatureExtractor
# from aldiscore.prediction.predictor import DifficultyPredictor
# from aldiscore.datastructures.ensemble import Ensemble
# from aldiscore.scoring import pairwise, set_based
import pathlib
from rich_argparse import RichHelpFormatter, RawTextRichHelpFormatter
from rich.table import Table
from rich.console import Console
from rich.markdown import Markdown
from io import StringIO

# from Bio.AlignIO import read

# from aldiscore.scoring import pairwise
# from aldiscore.scoring import set_based
# from aldiscore.datastructures.ensemble import Ensemble
# from aldiscore.datastructures.alignment import Alignment

# CLI structure: aldiscore (base command), then sub-command, then options
# Pairwise mode (input: folder with alternative MSAs in FASTA format)
# aldiscore heuristic <input_dir> [--strategy={d_ssp, d_seq, d_pos, d_phash}] [--format={scalar, flat, matrix}]
# default: aldiscore heuristic <input_dir> --strategy=d_pos --format=scalar

# Set-based mode (input: folder with alternative MSAs in FASTA format)
# aldiscore heuristic <input_dir> [--strategy={conf_set, conf_entropy, conf_displace}] [--format={scalar, sequence, residue}]
# default: aldiscore heuristic <input_dir> --strategy=conf_entropy --format=scalar

# Prediction mode (input: unaligned FASTA file)
# aldiscore predict <input_file> [--model={latest, vX.Y}] [--seed={1,2,3...}]
# default: aldiscore predict <input_file> --model=latest --seed=42

console = Console()

heuristics = ["pairwise", "set-based"]
strategy_map = dict(
    zip(
        heuristics,
        [
            ["d_ssp", "d_seq", "d_pos", "d_phash"],
            ["conf_set", "conf_entropy", "conf_displace"],
        ],
    )
)
format_map = dict(
    zip(
        heuristics,
        [
            ["scalar", "flat", "matrix"],
            ["scalar", "sequence", "residue"],
        ],
    )
)


def handle_heuristic_mode(input_dir, strategy, output_format):
    """
    Handles heuristic mode (pairwise or set_based) depending on strategy.
    """

    for mode in heuristics:
        strategy_valid = strategy in strategy_map[mode]
        format_valid = output_format in format_map[mode]
        if strategy_valid:
            break
    if not strategy_valid:
        raise ValueError(
            f"Unknown strategy: {strategy}. Choose from {str(strategy_map)}."
        )
    if not format_valid:
        raise ValueError(
            f"Invalid format '{output_format}' for strategy '{strategy}'. "
            f"Valid formats: {', '.join(format_map[strategy])}"
        )

    print(mode)
    print(strategy, input_dir, output_format)
    # call pairwise handler


def handle_predict_mode(input_path, model, seed):
    """
    Handles prediction mode for predicting alignment difficulty.

    :param input_path: Path to the unaligned FASTA file.
    :param model: The model version to use for prediction.
    :param seed: The random seed for prediction.
    """
    print("Predict:")
    print(input_path, model, seed)
    pass


def main():
    parser = argparse.ArgumentParser(
        description="A command-line tool for alignment difficulty prediction and scoring.",
        formatter_class=RichHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Heuristic (covers pairwise + set-based)
    heuristic_parser = subparsers.add_parser(
        "heuristic",
        help="Compute alignment dispersion for an ensemble of alternative MSAs.",
        formatter_class=RawTextRichHelpFormatter,
    )
    heuristic_parser.add_argument(
        "input_dir",
        type=pathlib.Path,
        help="Path to input directory containing FASTA files of alternative MSAs.",
    )
    heuristic_parser.add_argument(
        "--strategy",
        type=str,
        default="d_pos",
        help=(
            "Scoring strategy.\n"
            "Pairwise: {d_ssp, d_seq, d_pos, d_phash}\n"
            "Set-based: {conf_set, conf_entropy, conf_displace}"
        ),
    )
    heuristic_parser.add_argument(
        "--format",
        type=str,
        default="scalar",
        help="\n".join(
            (
                "Output format.",
                "Pairwise: {scalar, flat, matrix}",
                "Set-based: {scalar, sequence, residue}",
                "Defaults to scalar.",
            )
        ),
    )

    # Prediction
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict alignment dispersion for a single (unaligned) FASTA file.",
        formatter_class=RichHelpFormatter,
    )
    predict_parser.add_argument("input_path", type=pathlib.Path)
    predict_parser.add_argument(
        "--model",
        type=str,
        default="latest",
        help="Pretrained model version. Defaults to 'latest'.",
    )
    predict_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for sampling in randomized features. Defaults to 42.",
    )

    args = parser.parse_args()

    if args.command == "heuristic":
        handle_heuristic_mode(args.input_dir, args.strategy, args.format)
    elif args.command == "predict":
        handle_predict_mode(args.input_path, args.model, args.seed)

    args = parser.parse_args()


if __name__ == "__main__":
    main()
