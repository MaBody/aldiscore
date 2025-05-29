import enum
import pandas as pd
from pathlib import Path
import itertools
import numpy as np


class StringEnum(enum.Enum):
    def __str__(self):
        return self.value

    def __add__(self, other):
        return str(self) + other

    def __radd__(self, other):
        return other + str(self)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


class ExtendedEnum(StringEnum):
    def __new__(cls, value, description):
        # Create the Enum member instance
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj


class FeatureDirectionEnum(ExtendedEnum):
    """The directionality of the values of a feature.
    - ascending: bad/distant = 0, good/similar = 1
    - descending: bad/distant = 1, good/similar = 0
    """

    ASCENDING = ("ascending", "bad/distant = 0, good/similar = 1")
    DESCENDING = ("descending", "bad/distant = 1, good/similar = 0")


class FeatureTypeEnum(ExtendedEnum):
    """The general type of a feature.
    - quality: Quality metric computed on a single MSA.
    - distance: Distance metric computed across multiple MSAs.
    - other: Not relevant for the label generation process.
    """

    QUALITY = ("quality", "Quality metric computed on a single MSA.")
    DISTANCE = ("distance", "Distance metric computed across multiple MSAs.")
    OTHER = ("other", "Not relevant for the label generation process.")


class FeatureEnum(StringEnum):
    """Taxonomy of the features computed on MSA aligner output.

    Parameters
    ----------
    value : str
        Name/identifier of the feature.
    is_bounded : bool
        Whether the feature is bounded in the 0-1 range
    type : FeatureTypeEnum
        Whether the feature is a distance or a quality metric (or other).
    direction : FeatureDirectionEnum
        Whether the feature is ascending or descending in its values.
    """

    def __new__(
        cls,
        value: str,
        pretty: str,
        feat_type: FeatureTypeEnum,
        direction: FeatureDirectionEnum,
    ):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.pretty = pretty
        obj.feat_type = feat_type
        obj.direction = direction
        return obj

    @classmethod
    def extract_from_df(
        cls,
        df: pd.DataFrame,
        feature_enum=None,
        type_enum: FeatureTypeEnum = None,
        direction_enum: FeatureDirectionEnum = None,
        statistic_filter: str = None,
        filter_is_strict: bool = True,
        keep_string_cols: bool = True,
    ):
        """Extracts all columns from the df which have the specified type and/or direction.
           Either type_enum or direction_enum need to be specified.


        Parameters
        ----------
        df : pd.DataFrame
            DataFrame from which to extract the columns
        feature_enum : FeatureEnum = None
            Optional. Returns columns that are associated with the specified feature.
        type_enum : FeatureTypeEnum = None
            Optional. Returns columns of features which have the specified type.
        direction_enum : FeatureDirectionEnum = None
            Optional. Returns columns of features which have the specified direction.
        statistic_filter : str = None
            Optional. Additionally filter based for the specified statistic (e.g. "mean", "std")
        filter_is_strict : bool = True
            Optional. Only applied if statistic_filter is provided. If False, features like "dispersion" for which we do not compute multiple statistics are included in results anyway.
        """

        # TODO: add option for performance counters, remove them otherwise

        if feature_enum:
            feature_cols = set(df.columns[df.columns.str.contains(str(feature_enum))])
            if statistic_filter:
                stat_cols = set(df.columns[df.columns.str.startswith(statistic_filter)])
                feature_cols = feature_cols.intersection(stat_cols)
            if keep_string_cols:
                str_cols = set(df.select_dtypes(include=["object", "string"]).columns)
                feature_cols = feature_cols.union(str_cols)
            return df[list(feature_cols)].copy()

        matching_type = set()
        matching_direction = set()
        if type_enum:
            matching_type.update([e for e in cls if e.feat_type == type_enum])
        if direction_enum:
            matching_direction.update([e for e in cls if e.direction == direction_enum])

        matches = set()
        if matching_type and matching_direction:
            matches = matching_type.intersection(matching_direction)
        else:
            matches.update(matching_type)
            matches.update(matching_direction)
        if len(matches) == 0:
            return df[[]]
        matches = [e.value for e in matches]

        all_cols = pd.Series(df.columns)
        selected_cols = set()
        for substring in matches:
            substring_cols = all_cols[all_cols.str.contains(substring)]
            # Filter for a certain statistic if specified
            apply_filter = filter_is_strict | (len(substring_cols) > 1)
            if (statistic_filter is not None) & apply_filter:
                substring_cols = substring_cols[
                    substring_cols.str.startswith(statistic_filter)
                ]
            selected_cols.update(substring_cols)
            if keep_string_cols:
                selected_cols.update(df.select_dtypes(include=["object"]).columns)
        return df[list(selected_cols)].copy()

    # # # # # Distance metrics # # # # #

    COLUMN_CONFIDENCE = (
        "column_confidence",
        "Column Confidence",
        FeatureTypeEnum.DISTANCE,
        FeatureDirectionEnum.ASCENDING,
    )
    # LETTER_CONFIDENCE = (
    #     "letter_confidence",
    #     True,
    #     FeatureTypeEnum.DISTANCE,
    #     FeatureDirectionEnum.DESCENDING, # TODO: check if really descending like the data suggests
    # )
    DISPERSION = (
        "dispersion",
        "Dispersion",
        FeatureTypeEnum.DISTANCE,
        FeatureDirectionEnum.DESCENDING,
    )
    PERC_HASH_HAMMING = (
        "perc_hash_hamming_dist",
        "pHash",
        FeatureTypeEnum.DISTANCE,
        FeatureDirectionEnum.DESCENDING,
    )

    SSP_DIST = (
        "symmetrized_sp_dist",
        r"$d_{SSP}$",
        FeatureTypeEnum.DISTANCE,
        FeatureDirectionEnum.DESCENDING,
    )

    HOMOLOGY_SEQ_DIST = (
        "homology_seq_dist",
        r"$d_{seq}$",
        FeatureTypeEnum.DISTANCE,
        FeatureDirectionEnum.DESCENDING,
    )
    HOMOLOGY_POS_DIST = (
        "homology_pos_dist",
        r"$d_{pos}$",
        FeatureTypeEnum.DISTANCE,
        FeatureDirectionEnum.DESCENDING,
    )

    DISPLACEMENT_DIST = (
        "displacement_dist",
        "Displacement",
        FeatureTypeEnum.DISTANCE,
        FeatureDirectionEnum.DESCENDING,
    )

    GAP_DELTA_DISTANCE = (
        "gap_delta_dist",
        "Gap Delta",
        FeatureTypeEnum.DISTANCE,
        FeatureDirectionEnum.DESCENDING,
    )

    CONFUSION_SET = (
        "confusion_set",
        r"$\text{Conf}_{Set}$",
        FeatureTypeEnum.DISTANCE,
        FeatureDirectionEnum.DESCENDING,
    )
    CONFUSION_ENTROPY = (
        "confusion_entropy",
        r"$\text{Conf}_{Entropy}$",
        FeatureTypeEnum.DISTANCE,
        FeatureDirectionEnum.DESCENDING,
    )
    CONFUSION_DISPLACEMENT = (
        "confusion_displacement",
        r"$\text{Conf}_{Displace}$",
        FeatureTypeEnum.DISTANCE,
        FeatureDirectionEnum.DESCENDING,
    )

    # # # # # Quality metrics # # # # #

    INFLATION_RATIO = (
        "inflation_ratio",
        "",
        FeatureTypeEnum.QUALITY,
        FeatureDirectionEnum.ASCENDING,
    )
    INTRINSIC_SUM_OF_PAIRS = (
        "intrinsic_sp_score",
        "",
        FeatureTypeEnum.QUALITY,
        FeatureDirectionEnum.ASCENDING,
    )
    INTRINSIC_COLUMN_IDENTITY = (
        "intrinsic_column_identity",
        "",
        FeatureTypeEnum.QUALITY,
        FeatureDirectionEnum.ASCENDING,
    )

    # # # # # Other # # # # #

    PYTHIA_SCORE = (
        "pythia_score",
        "Pythia Score",
        FeatureTypeEnum.OTHER,
        FeatureDirectionEnum.DESCENDING,
    )


_CONFUSION_ENUMS = [
    FeatureEnum.CONFUSION_SET,
    FeatureEnum.CONFUSION_ENTROPY,
    FeatureEnum.CONFUSION_DISPLACEMENT,
]

# # # # # # # Positional Encodings # # # # # #


class PositionalEncodingEnum(StringEnum):
    UNIFORM = "uniform"  # encode all gaps with "-1"
    SEQUENCE = "sequence"  # encode all gaps in a sequence identically
    POSITION = "position"  # encode each gap region in a sequence with the index of site to the left
    RAW = "raw"  # use the raw characters as the encoding
    GAP_REGIONS = "gap_regions"  # encode all gap characters with the length of the gapped region they belong to
    # POSITION_ABSOLUTE = "position_abs"  # encode each gap region in a sequence with its "gap-only" index (independent of the site indices)


# # # # # # # Data Types # # # # # # #


class DataTypeEnum(StringEnum):
    DNA = "DNA"
    AA = "AA"


# # # # # # # Aligners # # # # # # #

SPACE_CHAR = "§"


class AlignerEnum(StringEnum):

    def __new__(cls, value: str, cmd: str, path: Path):
        # Create the Enum member instance
        obj = object.__new__(cls)
        obj._value_ = value
        obj.cmd = cmd.strip(" ").replace(" ", SPACE_CHAR)
        obj.path = path
        return obj

    # MAFFT has '--thread xxx' option
    MAFFT = (
        "Mafft",
        "{aligner} {options} --thread {threads} {input}",
        Path("/hits/fast/cme/bodynems/tools/mafft/mafft.bat"),
    )
    # ClustalO has '--threads=xxx' option
    CLUSTALO = (
        "ClustalO",
        "{aligner} -i {input} -o {output} -v --force {options} --threads={threads}",
        Path("/hits/fast/cme/bodynems/tools/clustalo-1.2.4-Ubuntu-x86_64"),
    )
    # Muscle3 does not support threads
    MUSCLE3 = (
        "Muscle3",
        "{aligner} {options} -in {input} -out {output}",
        Path("/hits/fast/cme/bodynems/tools/muscle3.8.31_i86linux64"),
    )
    # Muscle5 has '-threads xxx' option (default is 20!)
    MUSCLE5 = (
        "Muscle5",
        "{aligner} -threads {threads} -stratified {options} -align {input} -output {output}",
        Path("/hits/fast/cme/bodynems/tools/muscle5.1.linux_intel64"),
        # Path("/home/bodynems/tools/muscle5.1.linux_intel64"),
    )


# TODO: add aligner information to enum: list of [progressive, consistency, ]

op_default = 1.53
ep_default = 0.123
sigma = 0.25
sigma_range = np.array([1 - sigma, 1 + sigma])

op_range = list(np.linspace(*tuple(op_default * sigma_range), num=4).round(4))
ep_range = list(np.linspace(*tuple(ep_default * sigma_range), num=2).round(4))


class ToolEnum(StringEnum):

    def __new__(cls, value: str, options: str, aligner: AlignerEnum, max_n_taxa: int):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.options = options
        obj.aligner = aligner
        obj.max_n_taxa = max_n_taxa
        return obj

    @classmethod
    def get_option_variants(cls, tool_enum):
        """Returns a list with the cartesian product of all combinations of options, formatted as strings."""
        variants = []
        if tool_enum.options is None:
            variants.append("")
        else:
            base_options = []
            product_options = {}
            for key, val in tool_enum.options.items():
                if val is None:
                    base_options.append(key)
                else:
                    if not isinstance(val, list):
                        raise ValueError(f"Expected list of params, got {val}")
                    product_options[key] = val
            product_keys = list(product_options.keys())
            for product_vals in itertools.product(*product_options.values()):
                variant = base_options.copy()
                for key, val in zip(product_keys, product_vals):
                    variant.extend([key, str(val)])
                variants.append(variant)
        variants = [SPACE_CHAR.join(variant) for variant in variants]
        return variants

    @classmethod
    def format_cmd(cls, tool_enum, kwargs):
        commands = []
        temp = tool_enum.aligner.cmd.format(**kwargs).split(SPACE_CHAR)
        for command in temp:
            if command.strip(" "):
                commands.append(command)
        return commands

    # MAFFT instances
    MAFFT_FFT_NS_2 = (
        "Mafft_FFT-NS-2",
        {
            "--op": op_range,
            "--ep": ep_range,
        },
        AlignerEnum.MAFFT,
        None,
    )
    MAFFT_L_INS_I = (
        "Mafft_L-INS-i",
        {
            "--localpair": None,
            "--maxiterate": [1000],
            "--op": op_range,
            "--ep": ep_range,
        },
        AlignerEnum.MAFFT,
        None,
    )
    MAFFT_G_INS_I = (
        "Mafft_G-INS-i",
        {
            "--globalpair": None,
            "--maxiterate": [1000],
            "--op": op_range,
            "--ep": ep_range,
        },
        AlignerEnum.MAFFT,
        None,
    )

    # ClustalO instances
    ClUSTALO = (
        "ClustalO",
        {
            "--iterations": [6],
            "--max-guidetree-iterations": [1, 2, 3, 4],
            "--max-hmm-iterations": [1, 2],
        },
        AlignerEnum.CLUSTALO,
        None,
    )

    MUSCLE5_ALIGN = (
        "Muscle5",
        {
            "-replicates": [2],
        },
        AlignerEnum.MUSCLE5,
        None,
    )

    # Muscle3 instances
    MUSCLE3 = (
        "Muscle3",
        {
            # "-gapopen": [-12.75, -12.25, -11.75, -11.25],
            # "-gapextend": [-1.5, -1.25, -1, -0.75],
            # "-diags1": ["no", "yes"], # TODO: if we need to combine flags, implement that somehow...
            # "-diags2": ["no", "yes"],
            "-weight1": ["clustalw", "none"],
            "-weight2": ["clustalw", "none"],
            "-objscore": ["spf", "xp"],
            # "-maxiters": [1, 2, 8, 16],
        },
        AlignerEnum.MUSCLE3,
        None,
    )
