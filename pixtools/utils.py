"""
General utility functions for doing pixels analysis.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import default_rng

rng = default_rng()


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

        if axes.ndim == 1:
            self.axes_flat = list(axes)
            self.to_label = axes[-1]
        else:
            self.axes_flat = [ax for dim in axes for ax in dim]
            self.to_label = axes[-1][0]

        self.legend = self.axes_flat[-1]

        # hide excess axes that fill the grid
        for i in range(len(data), len(self.axes_flat)):
            self.axes_flat[i].set_visible(False)


def save(path, fig=None, nosize=False):
    """
    Save a figure to the specified path. If a file extension is not part of the path
    name, it is saved as a PDF. The current figure is used, or a specified figure can be
    passed as fig=<figure>.
    """
    path = Path(path).expanduser()

    if not fig:
        fig = plt.gcf()

    if not nosize:
        fig.set_size_inches(10, 10)

    if not path.suffix:
        path = path.with_suffix('.pdf')

    if path.suffix == '.pdf':
        with PdfPages(path) as pdf:
            pdf.savefig(figure=fig, bbox_inches='tight', dpi=300)

    else:
        fig.savefig(path, dpi=1200)

    if len(path.parts) > 3:
        path = "/".join(path.parts[-3:])
    print("Figure saved to: ", path)


def confidence_interval(array, axis=0, samples=10000, size=25):
    """
    Compute the 95% confidence interval of some data in an array.

    Parameters
    ==========
    array : np.array
        The data, can have any number of dimensions.

    axis : int, optional
        The axis in which to compute CIs. This axis is collapsed in the output. Default:
        0.

    samples : int, optional
        The number of samples to bootstrap. Default: 10000.

    size : int, optional
        The size of each boostrapped sample. Default: 25.

    """
    samps = np.array(
        [rng.choice(array, size=size, axis=axis) for _ in range(samples)]
    )
    medians = np.median(samps, axis=-1)
    results = np.percentile(medians, [2.5, 97.5], axis=0)
    return results[1, ...] - results[0, ...]
