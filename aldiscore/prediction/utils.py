import numpy as np
from typing import Literal
import itertools
import random
import math


def sample_index_tuples(n: int, r: int, k: int):
    """
    This is necessary because itertools.combinations is inefficient for sampling + combinatorics.
    - n: index range
    - r: tuple length
    - k: (maximum) number of tuples
    """
    samples = set()
    done = False
    max_comb = math.comb(n, r)
    limit = np.minimum(max_comb, k)
    while not done:
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
