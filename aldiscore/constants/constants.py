import numpy as np
from collections import defaultdict

DNA_CHARS = np.array(list("ACGT"))
AA_CHARS = np.array(list("ACDEFGHIKLMNPQRSTVWY"))

GAP_CHAR = "-"
GAP_CONST = -1


DNA_CHAR_MAP = defaultdict(lambda: 0, {"A": 63, "C": 127, "G": 191, "T": 255})

_AA_CHARS_EXT = [char for char in "-ACDEFGHIKLMNOPQRSTUVWY"]
# Creating 23 numbers from 0 to 255 with maximal spacing
_AA_NUM = np.arange(len(_AA_CHARS_EXT)) / (len(_AA_CHARS_EXT) - 1) * 255
# Round to intergers (spacing of 11 or 12)
_AA_NUM = np.round(_AA_NUM, decimals=0).astype(int)
AA_CHAR_MAP = defaultdict(
    lambda: 0, {char: mapping for char, mapping in zip(_AA_CHARS_EXT, _AA_NUM)}
)
