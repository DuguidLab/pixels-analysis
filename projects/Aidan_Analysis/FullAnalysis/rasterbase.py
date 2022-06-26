#First import required packages from base. 


from base import * #Import our experiment and metadata needed
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from math import ceil

#Add the location of pixtools to the path
# sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis") #Adds the location of the pixtools folder to the path
# from pixtools.utils import Subplots2D #Allows us to use the custom class designed to give the figs/axes from a set of subplots that will fit the data. This generates as many subplots as required in a square a shape as possible

#Now import our sorted data functions


#Defines a function that will return plots of EITHER units per trial or trials per unit (i.e., inverted datasets)
#To do so, specify "trial" or "unit" as the per value
def _raster(per, data, sample, start, subplots, label):
    per_values = data.columns.get_level_values(per).unique()
    per_values.values.sort()

    if per == "trial":
        data = data.reorder_levels(["trial", "unit"], axis=1)
        other = "unit"
    else:
        other = "trial"

    # units if plotting per trial, trials if plotting per unit
    samp_values = list(data.columns.get_level_values(other).unique().values)
    if sample and sample < len(samp_values):
        sample = random.sample(samp_values, sample)
    else:
        sample = samp_values

    if subplots is None:
        subplots = Subplots2D(per_values)

    palette = sns.color_palette("pastel") #Set the colours for the graphs produced

    for i, value in enumerate(per_values):
        val_data = data[value][sample] #Requires I index into the data when assembling plots (see plots.py)
        val_data.columns = np.arange(len(sample)) + start
        val_data = val_data.stack().reset_index(level=1)

        p = sns.scatterplot(
            data=val_data,
            x=0,
            y='level_1',
            ax=subplots.axes_flat[i],
            s=0.5,
            legend=None,
        )
        p.set_yticks([])
        p.set_xticks([])
        p.autoscale(enable=True, tight=True)
        p.get_yaxis().get_label().set_visible(False)
        p.get_xaxis().get_label().set_visible(False)
        p.text(
            0.95, 0.95,
            value,
            horizontalalignment='right',
            verticalalignment='top',
            transform=subplots.axes_flat[i].transAxes,
            color='0.3',
        )
        p.axvline(c=palette[2], ls='--', linewidth=0.5)

        if not subplots.axes_flat[i].yaxis_inverted():
            subplots.axes_flat[i].invert_yaxis()

    if label: #Chane our axis labels, here to trial number and time from the reach (which the data is aligned to!)
        to_label = subplots.to_label
        to_label.get_yaxis().get_label().set_visible(True)
        to_label.set_ylabel('Trial Number')
        to_label.get_xaxis().get_label().set_visible(True)
        to_label.set_xlabel('Time from reach (s)')

        # plot legend subplot
        legend = subplots.legend
        legend.text(
            0, 0.9,
            'Trial number' if per == 'trial' else 'Unit ID',
            transform=legend.transAxes,
            color='0.3',
        )
        legend.set_visible(True)
        legend.get_xaxis().set_visible(False)
        legend.get_yaxis().set_visible(False)
        legend.set_facecolor('white')

    return subplots

#Using this function we may now plot the graphs we desire!
#First let us define a function to plot spikes per unit

def per_unit_raster(data, sample=None, start=0, subplots=None, label=True):
    """
    Plots a spike raster for every unit in the given data with trials as rows.

    data is a multi-dimensional dataframe as returned from
    Experiment.align_trials(<action>, <event>, 'spike_times') indexed into to get the
    session and recording.
    """
    return _raster("unit", data, sample, start, subplots, label)

#And per trial
def per_trial_raster(data, sample=None, start=0, subplots=None, label=True):
    """
    Plots a spike raster for every trial in the given data with units as rows.

    data is a multi-dimensional dataframe as returned from
    Experiment.align_trials(<action>, <event>, 'spike_times') indexed into to get the
    session and recording.
    """
    return _raster("trial", data, sample, start, subplots, label)

#Now finally define a function that allows plotting a single unit as a raster
def single_unit_raster(data, ax=None, sample=None, start=0, unit_id=None):
    # units if plotting per trial, trials if plotting per unit
    samp_values = list(data.columns.get_level_values('trial').unique().values)
    if sample and sample < len(samp_values):
        sample = random.sample(samp_values, sample)
    else:
        sample = samp_values

    val_data = data[sample]
    val_data.columns = np.arange(len(sample)) + start
    val_data = val_data.stack().reset_index(level=1)

    if not ax:
        ax = plt.gca()

    p = sns.scatterplot(
        data=val_data,
        x=0,
        y='level_1',
        ax=ax,
        s=1.5,
        legend=None,
        edgecolor=None,
    )
    p.set_yticks([])
    p.set_xticks([])
    p.autoscale(enable=True, tight=True)
    p.get_yaxis().get_label().set_visible(False)
    p.get_xaxis().get_label().set_visible(False)

    palette = sns.color_palette()
    p.axvline(c='black', ls='--', linewidth=0.5)

    if not ax.yaxis_inverted():
        ax.invert_yaxis()

    if unit_id is not None:
        p.text(
            0.95, 0.95,
            unit_id,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            color='0.3',
        )
