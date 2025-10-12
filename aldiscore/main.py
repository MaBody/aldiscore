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

out_format_map = {}
out_format_map["pairwise"] = ["scalar", "flat", "matrix"]
out_format_map["set-based"] = ["scalar", "sequence", "residue"]


def handle_heuristic_mode(in_dir, in_format, method, out_format):
    """
    Handles heuristic mode (pairwise or set_based) depending on method.
    """
    for mode in heuristic_modes:
        method_valid = method in mode_map[mode]
        is_valid_format = out_format in out_format_map[mode]
        if method_valid:
            break
    if not method_valid:
        raise ValueError(f"Unknown method: {method}. Choose from {str(mode_map)}.")
    if not is_valid_format:
        raise ValueError(
            f"Invalid output format '{out_format}' for method '{method}'. "
            f"Valid formats: {', '.join(out_format_map[mode])}"
        )

    from aldiscore.datastructures.ensemble import Ensemble

    # Load data (use placeholder value for in_type, not relevant for heuristics used below)
    ensemble = Ensemble.load(ensemble_dir=in_dir, in_format=in_format, in_type="DNA")

    if mode == "pairwise":

        from aldiscore.scoring import pairwise

        if method == ME.D_POS:
            return pairwise.DPosDistance(out_format).compute(ensemble)
        elif method == ME.D_SSP:
            return pairwise.SSPDistance(out_format).compute(ensemble)
        elif method == ME.D_SEQ:
            return pairwise.DSeqDistance(out_format).compute(ensemble)

    if mode == "set-based":

        from aldiscore.scoring import set_based

        if method == ME.CONF_ENTROPY:
            return set_based.ConfusionEntropy(out_format).compute(ensemble)
        elif method == ME.CONF_SET:
            return set_based.ConfusionSet(out_format).compute(ensemble)
        elif method == ME.CONF_DISPLACE:
            return set_based.ConfusionDisplace(out_format).compute(ensemble)


def handle_predict_mode(
    in_path, drop_gaps, in_format, in_type, max_samples, model, seed
):
    """
    Handles prediction mode for predicting alignment difficulty.

    :param in_path: Path to the sequence file.
    :param drop_gaps: Indicates whether gaps must be dropped from the sequences.
    :param in_format: File format of the sequences.
    :param in_type: Input data type (DNA, AA, auto).
    :param max_samples: Upper bound on number of sequence triplets.
    :param model: The model version to use for prediction.
    :param seed: The random seed for prediction.
    """
    from aldiscore.prediction.predictor import DifficultyPredictor

    predictor = DifficultyPredictor(model=model, max_samples=max_samples, seed=seed)
    return predictor.predict(
        sequences=in_path,
        in_format=in_format,
        in_type=in_type,
        drop_gaps=drop_gaps,
    )


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
        help="Path to input directory containing multiple MSA files.",
    )
    heuristic_parser.add_argument(
        "--in-format",
        type=str,
        default="fasta",
        help="File format of the input sequences. Defaults to 'fasta'. Must be supported by BioPython.",
    )
    heuristic_parser.add_argument(
        "--method",
        type=str,
        default="d_pos",
        help="\n".join(
            (
                "Scoring method. Defaults to 'd_pos'.",
                f"Pairwise:  {mode_map['pairwise']}",
                f"Set-based: {mode_map['set-based']}",
            )
        ),
    )
    heuristic_parser.add_argument(
        "--out-format",
        type=str,
        default="scalar",
        help="\n".join(
            (
                "Output format. Defaults to 'scalar'.",
                f"Pairwise:  {out_format_map['pairwise']}",
                f"Set-based: {out_format_map['set-based']}",
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
        "--drop-gaps",
        action="store_true",
        dest="drop_gaps",
        help="If set, gaps in the input sequences are dropped (set this flag for aligned input data).",
    )
    predict_parser.add_argument(
        "--in-format",
        type=str,
        default="fasta",
        help="File format of the input sequences. Defaults to 'fasta'. Must be supported by BioPython.",
    )
    predict_parser.add_argument(
        "--in-type",
        type=str,
        default="auto",
        help="Input data type. Choose from {'auto', 'DNA', 'AA'}. If set to 'auto', we use a heuristic to infer. Defaults to 'auto'.",
    )
    predict_parser.add_argument(
        "--max-samples",
        type=int,
        default=333,
        help="Upper bound on number of sequence triplets sampled for transitive consistency features. Trade-off between variance and compute. Defaults to 333.",
    )
    predict_parser.add_argument(
        "--model",
        type=str,
        default="latest",
        help="Indicates the pretrained model version. Either 'latest' or following the format 'vX.Y'.",
    )
    predict_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for sampling in randomized features. Defaults to 0.",
    )
    args = parser.parse_args()
    out = None
    try:
        if args.command == "heuristic":
            out = handle_heuristic_mode(
                in_dir=getattr(args, "in-dir"),
                in_format=args.in_format,
                method=args.method,
                out_format=args.out_format,
            )
        elif args.command == "predict":
            out = handle_predict_mode(
                in_path=getattr(args, "in-path"),
                drop_gaps=args.drop_gaps,
                in_format=args.in_format,
                in_type=args.in_type,
                max_samples=args.max_samples,
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
