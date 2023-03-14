import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pixtools.utils import Subplots2D, save


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


def plot_cell_type_waveforms(waveforms_sets, colours, align="trough"):
    """
    params
    ===
    waveform_sets: df, concat pd dataframe of waveforms.

    colours: dict, colour of each cell type; usually defined in style.py.

    align: str, alignment method.
        trough: aligh waveforms to trough (default).
        first_drop: normalise waveforms to the precedent traces, and align them at
            the peak before the first dropping point.
    """
    colours = list(colours.values())
    #names = list(waveforms_sets.columns.names)
    types = list(waveforms_sets.groupby(level="type", axis=1).groups.keys())
    types.sort(reverse=True)
    units_count = waveforms_sets.columns.get_level_values('unit').unique().shape[0]
    rolls = np.full((len(types), units_count), np.nan)
    centre = np.full(len(types), np.nan)

    means = []
    fig = plt.gcf()

    for t, cell_type in enumerate(types):
        waveforms = waveforms_sets[cell_type]
        # get means
        waveform_means = waveforms.groupby(level='unit', axis=1).mean()
        # find trough, same for all types
        #TODO: trough is np.argmin, not the middle
        assert 0
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


def plot_chronic_cell_type_waveforms(mouse_waveforms, colours, align="trough"):
    """
    Plots cell type waveforms for each mouse, from multiple sessions. 

    params
    ===
    mouse_waveforms: df, concat pd dataframe of waveforms from multiple sessions of a
    mouse.

    colours: dict, colour of each cell type.

    align: str, alignment method.
        trough: aligh waveforms to trough (default).
        first_drop: normalise waveforms to the precedent traces, and align them at
            the peak before the first dropping point.
    """
    #names = list(mouse_waveforms.columns.names)
    types = list(mouse_waveforms.groupby(level="type", axis=1).groups.keys())
    types.sort(reverse=False)
    sessions = mouse_waveforms.columns.get_level_values('session').unique()
    istep = mouse_waveforms.index[1]

    fig = plt.gcf()
    maxs = [-1, 1]

    cell_type_waveforms = {}

    for t, cell_type in enumerate(types):
        means = []
        troughs_ls = []

        # get waveforms of each session
        for session in sessions:
            session_waveforms = mouse_waveforms[session]

            # get waveforms of each cell type
            try:
                waveforms = session_waveforms[cell_type]
            except:
                print(f'> session {session} does not have {cell_type} units,\
                        \nnext sessions.\n')
                continue

            # get average spike waveform for each unit
            waveform_means = waveforms.groupby(level='unit', axis=1).mean()
            # add session index as prefix to allow concat
            waveform_means = waveform_means.add_prefix(f'{session}_')
            # get trough for each unit
            trough = waveform_means.idxmin()
            # append troughs from each session
            troughs_ls.append(trough)
            # append average waveforms from each session
            means.append(waveform_means)

        # concat all units of the same type across sessions
        cell_type_waveforms[cell_type] = pd.concat(means, axis=1)
        # concat all troughs
        troughs = pd.concat(troughs_ls)

        if align == "trough":
            # use the most common trough as centre
            centre = troughs.mode()[0]
            # normalise to trough
            norm_means = cell_type_waveforms[cell_type] / cell_type_waveforms[cell_type].abs().max()

            # get number of rolls
            n_roll = ((centre - troughs) / istep).round()
            # roll to align troughs
            for u in norm_means: # loop over all units
                norm_means[u] = np.roll(
                    norm_means[u].values,
                    int(n_roll[u]),
                )

            rolls = n_roll.values
            left_clip = int(rolls.max())
            right_clip = int(rolls.min())

            # make sure right clip is not smaller than left
            if right_clip > -1:
                right_clip = -1

            clipped = norm_means.iloc[left_clip : right_clip, :]
            clipped.index = clipped.index - centre

            plt.xlabel("Time to Trough (ms)")
            plt.ylabel("Ratio to Trough")
            plt.xlim((-1.5, 1.5))

        elif align == "precedent":
            assert 0
            #TODO: mar 13th not yet fixed! something wrong with the rolling
            # normalise to precedent trace
            pre = abs(cell_type_waveforms[cell_type].loc[:align, :].median())
            norm_means = cell_type_waveforms[cell_type] / pre

            # items in each cell type shares the same centre
            centre = troughs - 0.5
            align = centre.mode()[0]
            # only roll ones with longer spike extraction period
            n_roll = np.zeros(centre.shape[0])
            n_roll[np.where(centre > 1)[0]] = (centre[np.where(centre > 1)[0]] /
                                               istep).round()

            # roll to align troughs
            """
            for u in norm_means: # loop over all units
                cell_type_waveforms[cell_type]
                norm_means[u] = np.roll(
                    norm_means[u].values,
                    int(n_roll[u]),
                )
            """
            for u, unit in enumerate(cell_type_waveforms[cell_type]): # loop over all units
                cell_type_waveforms[cell_type][unit] = np.roll(
                    cell_type_waveforms[cell_type][unit].values,
                    int(n_roll[u]),
                )
            rolls = n_roll

            left_clip = int(rolls.max())
            right_clip = int(rolls.min())

            # make sure right clip is not smaller than left
            if right_clip > -1:
                right_clip = -1
            clipped = norm_means.iloc[left_clip : right_clip, :]
            clipped.index = clipped.index - align

            # find the maximum for ylim
            maxs.append(round(norm_means.median(axis=1).abs().max()))
            plt.xlabel("Time to Trough-0.5 (ms)")
            plt.ylabel("Ratio to Median of Precedent Trace")
            plt.xlim((-1, 3))

        # plot raw traces
        plt.plot(
            clipped,
            color=colours[cell_type],
            alpha=0.2,
            linewidth=0.5,
        )

        # plot median traces
        plt.plot(
            clipped.median(axis=1),
            color=colours[cell_type],
            alpha=0.8,
            linewidth=2,
            label=cell_type,
        )

    plt.legend()
    lims = [-max(maxs), max(maxs)]
    plt.ylim(lims)

    return fig
