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
    # add one so we always have a legend subplot
    s = math.sqrt(len(data) + 1)
    fig, axes = plt.subplots(round(s), math.ceil(s), *args, **kwargs)

    flat = [ax for dim in axes for ax in dim]

    # the bottom left plot is returned for axis labelling
    to_label = axes[-1][0]

    # hide unneeded remainder axes that pad square
    for i in range(len(data), len(flat)):
        flat[i].set_visible(False)

    if flatten:
        axes = flat

    return fig, axes, to_label
