import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pixtools.utils import Subplots2D, save
from npx.style import colours_cell_type


def session_waveforms(data, n=100):
    units = data.columns.get_level_values('unit').unique()

    subplots = Subplots2D(units)
    palette = sns.color_palette()

    for i, unit in enumerate(units):
        u_data = data[unit]
        if u_data.shape[1] > n:
            spikes = random.sample(list(u_data.columns.values), k=n)
            u_data = u_data[spikes]

        ax = subplots.axes_flat[i]
        p = sns.lineplot(
            data=u_data,
            ax=ax,
            legend=False,
            linewidth=0.5,
            alpha=0.1,
        )

        p.autoscale(enable=True, tight=True)
        p.set_yticks([])
        p.set_xticks([])
        p.text(
            0.95, 0.95,
            unit,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            color=palette[0],
        )
        p.get_yaxis().get_label().set_visible(False)
        p.get_xaxis().get_label().set_visible(False)
        p.set(facecolor='white')

    subplots.to_label.get_yaxis().get_label().set_visible(True)
    subplots.to_label.set_ylabel('unit?')
    subplots.to_label.get_xaxis().get_label().set_visible(True)
    subplots.to_label.set_xlabel('Time (ms)')
    subplots.to_label.set_xticks([data.index[0], data.index[-1]])

    return subplots


def cell_type_waveforms(waveforms_sets, align="trough"):
    """
    waveform_sets: df, concat pd dataframe of waveforms.
    align: str, alignment method.
        trough: aligh waveforms to trough (default).
        first_drop: normalise waveforms to the precedent traces, and align them at
            the peak before the first dropping point.
    """
    colours = list(colours_cell_type.values())
    names = list(waveforms_sets.columns.names)
    types = list(waveforms_sets.groupby(level="type", axis=1).groups.keys())
    types.sort(reverse=True)

    means = []
    units_count = waveforms_sets.columns.get_level_values('unit').unique().shape[0]
    rolls = np.full((len(types), units_count), np.nan)
    centre = np.full(len(types), np.nan)

    fig = plt.gcf()

    for t, cell_type in enumerate(types):
        waveforms = waveforms_sets[cell_type]
        # get means
        waveform_means = waveforms.groupby(level='unit', axis=1).mean()
        # find trough, same for all types
        trough = round(waveform_means.index[-1] - waveform_means.index[0]) / 2
        istep = waveform_means.index[1] - waveform_means.index[0]

        if align == "trough":
            # normalise to maximum, i.e., trough
            norm_means = waveform_means / waveform_means.abs().max()
            centre = trough

            # get number of rolls; 0.2 is arbitrary...
            n_roll = ((centre - norm_means.loc[1 : (centre + 0.2)].idxmin()) / istep).round()
            # roll to align troughs
            for u in norm_means: # loop over all units
                norm_means[u] = np.roll(
                    norm_means[u].values,
                    int(n_roll[u]),
                )
            # remove nan
            rolls[t, :n_roll.shape[0]] = n_roll.values

        elif align == "precedent":
            # items in each cell type shares the same centre
            centre = trough - 0.5
            
            pre = abs(waveform_means.loc[:centre, :].median())
            norm_means = waveform_means / pre
            norm_means.index = norm_means.index - centre 

        means.append(norm_means)

    #TODO: how to i avoid having two loops?
    maxs = [-1, 1]
    for d, df in enumerate(means):
        if align == "trough":
            rolls = rolls[~np.isnan(rolls)]
            left_clip = int(rolls.max())
            right_clip = int(rolls.min())
            norm_means = df.iloc[left_clip : right_clip, :]
            norm_means.index = norm_means.index - centre
            plt.xlabel("Time to Trough (ms)")
        elif align == "precedent":
            norm_means = df
            maxs.append(norm_means.median(axis=1).abs().max().round())
            plt.xlabel("Time to Trough-0.5 (ms)")
            plt.xlim((-1, 3))

        # plot raw traces
        plt.plot(
            norm_means,
            color=colours[d],
            alpha=0.2,
            linewidth=0.5,
        )
        # plot median traces
        plt.plot(
            norm_means.median(axis=1),
            color=colours[d],
            alpha=0.8,
            linewidth=2,
            label=types[d],
        )
    plt.legend()
    plt.ylabel("Ratio to Median of Precedent Trace")
    lims = [-max(maxs), max(maxs)]
    plt.ylim(lims)

    return fig
