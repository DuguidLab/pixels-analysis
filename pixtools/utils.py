"""
General utility functions for doing pixels analysis.
"""

import math
import matplotlib.pyplot as plt


def subplots2d(data, flatten=True, *args, **kwargs):
    """
    This will give you the `fig, axes` output from a 2D spread of subplots that fits
    your data. `flatten` will flatten the list of lists of axis objects returned by
    plt.subplots so that it can more easily be iterated over.
    """
    s = math.sqrt(len(data))
    fig, axes = plt.subplots(round(s), math.ceil(s), *args, **kwargs)

    if flatten:
        axes = [ax for dim in axes for ax in dim]

    return fig, axes
