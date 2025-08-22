import numpy as np
from typing import Literal
import itertools
import random
import math
from aldiscore.scoring.encoding import _ffill_numpy_2d_axis_1


def sample_index_tuples(n: int, r: int, k: int):
    """
    This is necessary because itertools.combinations is inefficient for sampling + combinatorics.
    - n: index range
    - r: tuple length
    - k: (maximum) number of tuples
    """
    samples = set()
    max_comb = math.comb(n, r)
    limit = np.minimum(max_comb, k)
    while True:
        # Sample k r-tuples randomly -> (k,r)
        samples_new = np.sort(
            np.random.randint(low=0, high=n, size=2 * k * r).reshape(2 * k, r), axis=1
        )
        # Remove rows where two or more positions are identical
        diff_mask = (np.diff(samples_new, axis=1) != 0).all(axis=1)
        samples_new = samples_new[diff_mask].tolist()
        # print(samples_new)
        samples_new = list(map(lambda tup: tuple(sorted(tup)), samples_new))
        samples.update(samples_new)
        if len(samples) >= limit:
            break

    return list(samples)[:k]


def compute_gap_lengths(alignment: np.ndarray, gap_code) -> np.ndarray:
    """Computes the gap lengths in a given alignment. Returns a flat array."""
    gap_mask = alignment == gap_code
    gap_ends = np.diff(gap_mask, n=1, axis=1, append=0) == -1
    # gap_counts contains an incrementing counter for gapped positions in each sequence
    gap_counts = gap_mask.cumsum(axis=1)
    # Set all positions to zero that are not the end of a gap region
    gap_counts[~gap_ends] = 0
    # For each gap region, the last gap now holds the count of all gaps up to that point
    # Right-shift the gap end counters (counts[i] = counts_shifted[i+1])
    gap_end_counts_shifted = np.roll(gap_counts, shift=1, axis=1)
    # Set the first position to zero (no preceding gap regions)
    gap_end_counts_shifted[:, 0] = 0
    # Forward fill the values: now position i holds the cumulative gap count prior to the current gap region
    gap_end_counts_shifted = _ffill_numpy_2d_axis_1(gap_end_counts_shifted, val=0)
    # Compute the length of the gap regions at their last position
    site_codes = gap_counts - gap_end_counts_shifted
    # Remove all positions that are not gap region ends
    site_codes[~gap_ends] = 0
    return site_codes[site_codes > 0]
