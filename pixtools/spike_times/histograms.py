"""
Plot a histogram of spike times around a specified event.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import ceil

from pixtools.utils import Subplots2D


def _histogram(level, data, session, bin_ms, duration, sample_rate):
    data = data[session]
    values = data.columns.get_level_values(level).unique()
    values.values.sort()

    if level == "trial":
        data = data.reorder_levels(["trial", "unit"], axis=1)

    # no. units if plotting per trial, no. trials if plotting per unit
    num_samples = len(data[values[0]].columns)

    subplots = Subplots2D(values)
    palette = sns.color_palette()

    for i, value in enumerate(values):
        val_data = data[value]
        val_data = val_data.values.reshape([val_data.values.size])
        val_data = val_data[np.logical_not(np.isnan(val_data))]
        ax = subplots.axes_flat[i]

        p = sns.histplot(
            val_data,
            binwidth=sample_rate * bin_ms / 1000,
            ax=ax,
            element="poly",
            linewidth=0,
        )
        p.set_yticks([])
        p.set_xticks([])
        p.autoscale(enable=True, tight=True)
        p.set_xlim([ - sample_rate * duration / 2, sample_rate * duration / 2])
        _, count = p.get_ylim()
        p.text(
            0.05, 0.95,
            "%.01f" % ((count / num_samples) / (bin_ms / 1000)),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            color='grey',
        )
        p.text(
            0.95, 0.95,
            value,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            color=palette[0],
        )
        p.get_yaxis().get_label().set_visible(False)
        p.get_xaxis().get_label().set_visible(False)
        p.axvline(c=palette[1], ls='--', linewidth=0.5)

    to_label = subplots.to_label
    to_label.get_yaxis().get_label().set_visible(True)
    to_label.set_ylabel('Firing rate (Hz)')
    to_label.get_xaxis().get_label().set_visible(True)
    to_label.set_xlabel('Time from push (s)')
    to_label.set_xticks([- sample_rate * duration / 2, 0, sample_rate * duration / 2])
    to_label.set_xticklabels([- duration / 2, 0,  duration / 2])

    # plot legend subplot
    legend = subplots.legend
    legend.text(
        0, 0.7,
        'Peak firing rate (Hz)',
        transform=legend.transAxes,
        color='grey',
    )
    legend.text(
        0, 0.3,
        'Trial number' if level == 'trial' else 'Unit ID',
        transform=legend.transAxes,
        color=palette[0],
    )
    legend.set_visible(True)
    legend.get_xaxis().set_visible(False)
    legend.get_yaxis().set_visible(False)
    legend.set_facecolor('white')

    return fig


def across_trials_histogram(data, session, bin_ms=100, duration=1, sample_rate=30000):
    """
    Plots a histogram for every unit in the given session showing spike.

    data is a multi-dimensional dataframe as returned from Experiment.align_trials.
    """
    return _histogram("unit", data, session, bin_ms, duration, sample_rate)


def across_units_histogram(data, session, bin_ms=100, duration=1, sample_rate=30000):
    """
    Plots a histogram for every unit in the given session showing spike.

    data is a multi-dimensional dataframe as returned from Experiment.align_trials.
    """
    return _histogram("trial", data, session, bin_ms, duration, sample_rate)
