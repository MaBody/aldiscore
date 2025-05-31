import enum


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


# # # # # # # Positional Encodings # # # # # #


class PositionalEncodingEnum(StringEnum):
    UNIFORM = "uniform"  # encode all gaps with "-1"
    SEQUENCE = "sequence"  # encode all gaps in a sequence identically
    POSITION = "position"  # encode each gap region in a sequence with the index of site to the left
    RAW = "raw"  # use the raw characters as the encoding
    GAP_REGIONS = "gap_regions"  # encode all gap characters with the length of the gapped region they belong to


# # # # # # # Data Types # # # # # # #


class DataTypeEnum(StringEnum):
    DNA = "DNA"
    AA = "AA"

    # # # # # Scoring Methods # # # # #


class FeatureEnum(StringEnum):
    """Taxonomy of the features computed on MSA aligner output.

    Parameters
    ----------
    value : str
        Name/identifier of the feature.
    pretty : str
        LaTeX-like name for plotting.
    """

    def __new__(
        cls,
        value: str,
        pretty: str,
    ):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.pretty = pretty
        return obj

    PERC_HASH_HAMMING = ("pHash", "pHash")

    SSP_DIST = ("d_SSP", r"$d_{SSP}$")
    D_SEQ_DIST = ("d_seq", r"$d_{seq}$")
    D_POS_DIST = ("d_pos", r"$d_{pos}$")

    CONFUSION_SET = ("ConfSet", r"$\text{Conf}_{Set}$")
    CONFUSION_ENTROPY = ("ConfEntropy", r"$\text{Conf}_{Entropy}$")
    CONFUSION_DISPLACE = ("ConfDisplace", r"$\text{Conf}_{Displace}$")
