"""
Plot a raster of spike times around a specified event.
"""


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from operator import itemgetter
from math import ceil

from pixtools.utils import Subplots2D

try:
    from itertools import pairwise
except:
    from itertools import tee

    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


def _raster(
    per,
    data,
    sample=None,  # Randomly sample this many trials/units
    start=0,  # Start plotting from this Y axis. Useful for multiple calls.
    subplots=None,  # Plot onto this Subplots2D, otherwise make some.
    label=True,  # Write the trial/unit ID in the plot corner.
    sortby=None,  # Sort trials by this list. A list of tuples that are used to draw patches.
    more_patches=None,  # Another list of tuples, only used for drawing more patches.
    size=1,  # Marker size.
    c=None,  # Marker colour.
    patch_colors=None,  # Patch colours.
    more_colors=None,  # Patch colours for the 'more_patches' patches.
):
    per_values = data.columns.get_level_values(per).unique()
    per_values.values.sort()

    if per == "trial":
        data = data.reorder_levels(["trial", "unit"], axis=1)
        other = "unit"
    else:
        other = "trial"

    # units if plotting per trial, trials if plotting per unit
    samp_values = list(data.columns.get_level_values(other).unique().values)
    if sortby is not None:
        assert len(sortby) == len(samp_values)

    # take sample, ensuring that we are still using the sortby values correctly
    if sample and sample < len(samp_values):
        assert not more_patches, "not yet implemented"
        if sortby:
            sortby, to_use = zip(
                *sorted(random.sample(sorted(zip(sortby, samp_values)), sample))
            )
            sortby = list(sortby)  # gives us tuples, which we can use for df indexing
            to_use = list(to_use)
        else:
            to_use = sorted(random.sample(samp_values, sample))

    else:
        to_use = samp_values
        if sortby is not None:
            sort_order = np.argsort(list(map(itemgetter(0), sortby)))
            sortby = np.array(sortby)[sort_order]
            to_use = np.array(to_use)[sort_order]
            if more_patches:
                more_patches = np.array(more_patches)[sort_order]

    if subplots is None:
        subplots = Subplots2D(per_values, sharex=True, sharey=True)

    if c is None:
        palette = sns.color_palette()
        c = palette[0]

    if not patch_colors:
        patch_colors = sns.color_palette()

    if not more_colors:
        more_colors = patch_colors

    for i, value in enumerate(per_values):
        val_data = data[value][to_use]
        val_data.columns = np.arange(len(to_use)) + start
        val_data = val_data.stack().reset_index(level=1)

        ax = subplots.axes_flat[i]
        ax.scatter(
            x=val_data[0],
            y=val_data["level_1"],
            s=size,
            linewidths=size * 0.5,
            facecolors="none",
            edgecolors=c,
            alpha=0.9,
        )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.autoscale(enable=True, tight=True)
        ax.get_yaxis().get_label().set_visible(False)
        ax.get_xaxis().get_label().set_visible(False)
        ax.text(
            0.95,
            0.95,
            value,
            horizontalalignment="right",
            verticalalignment="top",
            transform=subplots.axes_flat[i].transAxes,
            color="0.3",
        )
        ax.axvline(c="red", ls="--", linewidth=0.5)

        for p, (patches, colors) in enumerate(
            zip([sortby, more_patches], [patch_colors, more_colors])
        ):
            if patches is None:
                continue

            patch_coords = pd.concat(
                [
                    pd.DataFrame(np.arange(len(to_use)) + start).rename(
                        {0: "y"}, axis=1
                    ),
                    pd.DataFrame(patches),
                ],
                axis=1,
            )

            for col, (x1, x2) in enumerate(pairwise(patch_coords.columns[1:])):
                ax.fill_betweenx(
                    patch_coords["y"],
                    patch_coords[x1],
                    patch_coords[x2],
                    facecolor=colors[col],
                    linewidths=0,
                    alpha=0.6 - p * 0.2,
                    step="mid",
                    zorder=-2,
                )

        if not subplots.axes_flat[i].yaxis_inverted():
            subplots.axes_flat[i].invert_yaxis()

    if label:
        to_label = subplots.to_label
        to_label.get_yaxis().get_label().set_visible(True)
        to_label.set_ylabel("Trial")
        to_label.get_xaxis().get_label().set_visible(True)
        to_label.set_xlabel("Time from push (s)")

        # plot legend subplot
        legend = subplots.legend
        legend.text(
            0,
            0.9,
            "Trial number" if per == "trial" else "Unit ID",
            transform=legend.transAxes,
            color="0.3",
        )
        legend.set_visible(True)
        legend.get_xaxis().set_visible(False)
        legend.get_yaxis().set_visible(False)

    return subplots


def per_unit_raster(data, **kwargs):
    """
    Plots a spike raster for every unit in the given data with trials as rows.

    data is a multi-dimensional dataframe as returned from
    Experiment.align_trials(<action>, <event>, 'spike_times') indexed into to get the
    session and recording.
    """
    return _raster("unit", data, **kwargs)


def per_trial_raster(data, **kwargs):
    """
    Plots a spike raster for every trial in the given data with units as rows.

    data is a multi-dimensional dataframe as returned from
    Experiment.align_trials(<action>, <event>, 'spike_times') indexed into to get the
    session and recording.
    """
    return _raster("trial", data, **kwargs)


def single_unit_raster(
    data,
    ax=None,
    sample=None,
    start=0,
    unit_id=None,
    sortby=None,
    size=1,
    c=None,
    patch_colors=None,
):
    samp_values = list(data.columns.get_level_values("trial").unique().values)
    if sortby:
        assert len(sortby) == len(samp_values)

    if sample and sample < len(samp_values):
        to_use = sorted(random.sample(samp_values, sample))
        assert not sortby, "TODO"
    else:
        to_use = samp_values
        if sortby:
            sortby, to_use = zip(*sorted(zip(sortby, to_use)))
            sortby = list(sortby)
            to_use = list(to_use)

    val_data = data[to_use]
    val_data.columns = np.arange(len(to_use)) + start
    val_data = val_data.stack().reset_index(level=1).rename({"level_1": "trial"}, axis=1)

    if not ax:
        ax = plt.gca()

    ax.scatter(
        x=val_data[0],
        y=val_data["trial"],
        s=size,
        linewidths=size * 0.5,
        facecolors="none",
        edgecolors=c,
        alpha=0.9,
    )
    ax.set_yticks([])
    ax.set_xticks([])
    ax.autoscale(enable=True, tight=True)
    ax.get_yaxis().get_label().set_visible(False)
    ax.get_xaxis().get_label().set_visible(False)

    if not ax.yaxis_inverted():
        ax.invert_yaxis()

    if sortby:
        sortdata = pd.concat(
            [
                pd.DataFrame(np.arange(len(to_use)) + start).rename(
                    {0: "y"}, axis=1
                ),
                pd.DataFrame(sortby),
            ],
            axis=1,
        )

        if not patch_colors:
            patch_colors = sns.color_palette()

        for i, (x1, x2) in enumerate(pairwise(sortdata.columns[1:])):
            ax.fill_betweenx(
                sortdata["y"],
                sortdata[x1],
                sortdata[x2],
                facecolor=patch_colors[i],
                linewidths=0,
                alpha=0.6,
                step="mid",
                zorder=-2,
            )

    if unit_id is not None:
        ax.text(
            0.95,
            0.95,
            unit_id,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            color="0.3",
        )
