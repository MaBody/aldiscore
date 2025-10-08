import argparse
import pathlib
from rich_argparse import RichHelpFormatter, RawTextRichHelpFormatter
from rich.console import Console
import shutil
from aldiscore.enums.enums import MethodEnum as ME
import sys

console = Console()

heuristic_modes = ["pairwise", "set-based"]
mode_map = {}
mode_map["pairwise"] = list(map(str, [ME.D_SSP, ME.D_SEQ, ME.D_POS]))
mode_map["set-based"] = list(map(str, [ME.CONF_SET, ME.CONF_ENTROPY, ME.CONF_DISPLACE]))

out_type_map = {}
out_type_map["pairwise"] = ["scalar", "flat", "matrix"]
out_type_map["set-based"] = ["scalar", "sequence", "residue"]


def handle_heuristic_mode(in_dir, in_format, method, out_type):
    """
    Handles heuristic mode (pairwise or set_based) depending on method.
    """
    for mode in heuristic_modes:
        method_valid = method in mode_map[mode]
        out_type_valid = out_type in out_type_map[mode]
        if method_valid:
            break
    if not method_valid:
        raise ValueError(f"Unknown method: {method}. Choose from {str(mode_map)}.")
    if not out_type_valid:
        raise ValueError(
            f"Invalid output format '{out_type}' for method '{method}'. "
            f"Valid formats: {', '.join(out_type_map[mode])}"
        )

    from aldiscore.datastructures.ensemble import Ensemble

    # Load data
    ensemble = Ensemble.load(ensemble_dir=in_dir, in_format=in_format)

    if mode == "pairwise":

        from aldiscore.scoring import pairwise

        if method == ME.D_POS:
            return pairwise.DPosDistance(out_type).compute(ensemble)
        elif method == ME.D_SSP:
            return pairwise.SSPDistance(out_type).compute(ensemble)
        elif method == ME.D_SEQ:
            return pairwise.DSeqDistance(out_type).compute(ensemble)

    if mode == "set-based":

        from aldiscore.scoring import set_based

        if method == ME.CONF_ENTROPY:
            return set_based.ConfusionEntropy(out_type).compute(ensemble)
        elif method == ME.CONF_SET:
            return set_based.ConfusionSet(out_type).compute(ensemble)
        elif method == ME.CONF_DISPLACE:
            return set_based.ConfusionDisplace(out_type).compute(ensemble)


def handle_predict_mode(in_path, in_format, drop_gaps, model, seed):
    """
    Handles prediction mode for predicting alignment difficulty.

    :param in_path: Path to the sequence file.
    :param in_format: File format of the sequences.
    :param drop_gaps: Indicates whether gaps must be dropped from the sequences.
    :param model: The model version to use for prediction.
    :param seed: The random seed for prediction.
    """
    from aldiscore.prediction.predictor import DifficultyPredictor

    predictor = DifficultyPredictor(model, seed)
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
        "in-dir",
        type=pathlib.Path,
        help="Path to input directory containing multiple MSA files on the same sequences.",
    )
    heuristic_parser.add_argument(
        "--in-format",
        type=str,
        default="fasta",
        help="File format, defaults to 'fasta'.",
    )
    heuristic_parser.add_argument(
        "--method",
        type=str,
        default="d_pos",
        help="\n".join(
            (
                "Scoring method, defaults to 'd_pos'.",
                f"Pairwise:  {mode_map["pairwise"]}",
                f"Set-based: {mode_map["set-based"]}",
            )
        ),
    )
    heuristic_parser.add_argument(
        "--out-type",
        type=str,
        default="scalar",
        help="\n".join(
            (
                "Output format, defaults to 'scalar'.",
                f"Pairwise:  {out_type_map["pairwise"]}",
                f"Set-based: {out_type_map["set-based"]}",
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
        "in-path",
        type=pathlib.Path,
        help="Path to file containing multiple (unaligned) sequences.",
    )
    predict_parser.add_argument(
        "--in-format",
        type=str,
        default="fasta",
        help="Defaults to 'fasta'. File format of the input sequences. Must be supported by BioPython.",
    )
    predict_parser.add_argument(
        "--drop-gaps",
        action="store_true",
        dest="drop_gaps",
        help="If set, gaps in the input sequences are dropped (use for aligned input data).",
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
        default=0,
        help=" Defaults to 0. Seed used for sampling in randomized features.",
    )
    args = parser.parse_args()
    out = None
    try:
        if args.command == "heuristic":
            out = handle_heuristic_mode(
                in_dir=getattr(args, "in-dir"),
                in_format=args.in_format,
                method=args.method,
                out_type=args.out_type,
            )
        elif args.command == "predict":
            out = handle_predict_mode(
                in_path=getattr(args, "in-path"),
                in_format=args.in_format,
                drop_gaps=args.drop_gaps,
                model=args.model,
                seed=args.seed,
            )
    except:
        import traceback

        sys.stderr.write(traceback.format_exc())
        sys.exit(1)
    else:
        sys.stdout.write(str(out))
        sys.stdout.flush()
        sys.stderr.write("\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
