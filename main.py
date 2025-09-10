import argparse
import pathlib
from rich_argparse import RichHelpFormatter, RawTextRichHelpFormatter
from rich.console import Console
import shutil

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

heuristic_modes = ["pairwise", "set-based"]
strategy_map = dict(
    zip(
        heuristic_modes,
        [
            ["d_ssp", "d_seq", "d_pos", "d_phash"],
            ["conf_set", "conf_entropy", "conf_displace"],
        ],
    )
)
format_map = dict(
    zip(
        heuristic_modes,
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

    for mode in heuristic_modes:
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
            f"Invalid output format '{output_format}' for strategy '{strategy}'. "
            f"Valid formats: {', '.join(format_map[mode])}"
        )

    print(mode)
    print(strategy, input_dir, output_format)
    # call pairwise handler


def handle_predict_mode(in_path, in_format, gap_char, drop_gaps, model, seed):
    """
    Handles prediction mode for predicting alignment difficulty.

    :param in_path: Path to the sequence file.
    :param in_format: File format of the sequences.
    :param gap_char: Gap character if working with aligned sequences.
    :param drop_gaps: Indicates whether gaps must be dropped from the sequences.
    :param model: The model version to use for prediction.
    :param seed: The random seed for prediction.
    """
    from aldiscore.prediction.predictor import DifficultyPredictor

    predictor = DifficultyPredictor(model)
    return predictor.predict(in_path, in_format=in_format, drop_gaps=drop_gaps)


def get_formatter_cls(cls):
    formatter_class = lambda prog: cls(
        prog,
        max_help_position=40,  # indent before description starts
        width=shutil.get_terminal_size(
            (100, 20)
        ).columns,  # let it span full terminal width
    )
    return formatter_class


def main():

    parser = argparse.ArgumentParser(
        description="A command-line tool for alignment difficulty prediction and scoring.",
        formatter_class=get_formatter_cls(RichHelpFormatter),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Heuristic (covers pairwise + set-based)
    heuristic_parser = subparsers.add_parser(
        "heuristic",
        help="Compute alignment dispersion for an ensemble of alternative MSAs.",
        formatter_class=get_formatter_cls(RawTextRichHelpFormatter),
    )
    heuristic_parser.add_argument(
        "input_dir",
        type=pathlib.Path,
        help="Path to input directory containing multiple MSA files on the same sequences.",
    )
    heuristic_parser.add_argument(
        "--strategy",
        type=str,
        default="d_pos",
        help=(
            "Scoring strategy, defaults to 'd_pos'.\n"
            "Pairwise:  {d_ssp, d_seq, d_pos, d_phash}\n"
            "Set-based: {conf_set, conf_entropy, conf_displace}"
        ),
    )
    heuristic_parser.add_argument(
        "--out_format",
        type=str,
        default="scalar",
        help="\n".join(
            (
                "Output format, defaults to 'scalar'.",
                "Pairwise:  {scalar, flat, matrix}",
                "Set-based: {scalar, sequence, residue}",
                "Defaults to 'scalar'",
            )
        ),
    )

    # Prediction
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict alignment dispersion for an input file of multiple sequences.",
        formatter_class=get_formatter_cls(RichHelpFormatter),
    )
    predict_parser.add_argument(
        "in_path",
        type=pathlib.Path,
        help="Path to sequence file.",
    )
    predict_parser.add_argument(
        "--in_format",
        type=str,
        default="fasta",
        help="Defaults to 'fasta'. File format of the input sequences. Must be supported by BioPython.",
    )
    predict_parser.add_argument(
        "--drop_gaps",
        type=bool,
        default=True,
        help="Defaults to True. If True, gaps in the input sequences are removed (only necessary for aligned input data).",
    )
    predict_parser.add_argument(
        "--gap_char",
        type=str,
        default="-",
        help="Defaults to '-'. Gap character (only relevant when working with aligned sequences).",
    )
    predict_parser.add_argument(
        "--model",
        type=str,
        default="latest",
        help=" Defaults to 'latest'. Indicates the pretrained model version.",
    )
    predict_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=" Defaults to 42. Seed used for sampling in randomized features.",
    )
    args = parser.parse_args()

    if args.command == "heuristic":
        out = handle_heuristic_mode(
            args.in_path,
            args.in_format,
            args.gap_char,
            args.strategy,
            args.format,
        )
    elif args.command == "predict":
        out = handle_predict_mode(
            args.in_path,
            args.in_format,
            args.gap_char,
            args.drop_gaps,
            args.model,
            args.seed,
        )
    print(out)
    return out


if __name__ == "__main__":
    main()
