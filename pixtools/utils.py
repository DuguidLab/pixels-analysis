"""
General utility functions for doing pixels analysis.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt


class Subplots2D:
    """
    This will give you the `fig, axes` output from a 2D spread of subplots that fits
    your data.

    A number of attributes make it easy to access useful things:

     - legend : an extra subplot that can be used to create a legend
     - to_label : the bottom left subplot, for adding axis labels to
     - axes_flat : the axes but in a flatted array for easier iteration

    """
    def __init__(self, data, *args, **kwargs):
        s = math.sqrt(len(data) + 1)
        fig, axes = plt.subplots(round(s), math.ceil(s), *args, **kwargs)

        self.fig = fig
        self.axes = axes
        self.axes_flat = [ax for dim in axes for ax in dim]
        self.to_label = axes[-1][0]
        self.legend = self.axes_flat[len(data) + 1]

        # hide excess axes that fill the grid
        for i in range(len(data), len(self.axes_flat)):
            self.axes_flat[i].set_visible(False)


def save(path, fig=None):
    """
    Save a figure to the specified path in PDF form. The current figure is used, or a
    specified figure can be passed as fig=<figure>.
    """
    path = Path(path).expanduser().with_suffix('.pdf')
    savefig = fig.savefig if fig else plt.savefig

    with PdfPages(path) as pdf:
        savefig(bbox_inches='tight', dpi=300)
