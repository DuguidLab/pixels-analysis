"""
Plot a histogram of spike times around a specified event.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import ceil

from pixtools.utils import subplots2d


def _plot(level, data, ci, axes):
    values = data.columns.get_level_values(level).unique()
    values.values.sort()

    if level == "trial":
        data = data.reorder_levels(["trial", "unit"], axis=1)

    # no. units if plotting per trial, no. trials if plotting per unit
    num_samples = len(data[values[0]].columns)

    if axes is None:
        fig, axes, to_label = subplots2d(values, flatten=True)
    else:
        to_label = None
        fig = plt.gcf()

    palette = sns.color_palette()

    for i, value in enumerate(values):
        val_data = data[value].stack().reset_index()
        val_data['y'] = val_data[0]

        p = sns.lineplot(
            data=val_data,
            x='level_0',
            y='y',
            ci=ci,
            ax=axes[i],
            linewidth=0.5,
        )
        p.autoscale(enable=True, tight=True)
        p.set_yticks([])
        p.set_xticks([])
        peak = data[value].values.mean(axis=1).max()
        p.text(
            0.05, 0.95,
            "%.1f" % peak,
            horizontalalignment='left',
            verticalalignment='top',
            transform=axes[i].transAxes,
            color='grey',
        )
        p.text(
            0.95, 0.95,
            value,
            horizontalalignment='right',
            verticalalignment='top',
            transform=axes[i].transAxes,
            color=palette[0],
        )
        p.get_yaxis().get_label().set_visible(False)
        p.get_xaxis().get_label().set_visible(False)
        p.axvline(c=palette[1], ls='--', linewidth=0.5)

    if to_label:
        to_label.get_yaxis().get_label().set_visible(True)
        to_label.set_ylabel('Firing rate (Hz)')
        to_label.get_xaxis().get_label().set_visible(True)
        to_label.set_xlabel('Time from push (s)')
        to_label.set_xticks(data.index)
        #to_label.set_xticklabels([- duration / 2, 0,  duration / 2])

        # plot legend subplot
        legend = axes[i + 1]
        legend.text(
            0, 0.7,
            'Peak of mean',
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

    return fig, axes


def across_trials_plot(data, ci=95, axes=None):
    """
    Plots a histogram for every unit in the given session showing spike rate.

    Parameters
    ----------

    data : pandas.DataFrame
        A dataframe for a single session as returned from Behaviour.align_trials (or
        Experiment.align_trials but indexed into).

    ci : int or 'sd', optional
        Confidence intervals to plot as envelope around line. Default is 95 i.e. 95%
        confidence intervals. Also accepts "'sd'" which will draw standard deviation
        envelope, or None to plot no envelope.
    """
    return _plot("unit", data, ci, axes)


def across_units_plot(data, ci=95, axes=None):
    """
    Plots a histogram for every unit in the given session showing spike rate.

    Parameters
    ----------

    data : pandas.DataFrame
        A dataframe for a single session as returned from Behaviour.align_trials (or
        Experiment.align_trials but indexed into).

    ci : int or 'sd', optional
        Confidence intervals to plot as envelope around line. Default is 95 i.e. 95%
        confidence intervals. Also accepts "'sd'" which will draw standard deviation
        envelope, or None to plot no envelope.
    """
    return _plot("trial", data, ci, axes)
