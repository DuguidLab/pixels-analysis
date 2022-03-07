from base import *  # Import our experiment and metadata needed
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import probeinterface as pi
from reach import Session
from argparse import Action
from cv2 import bitwise_and
import matplotlib.pyplot as plt


sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")
from pixtools.utils import Subplots2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

    palette = sns.color_palette("pastel")  # Set the colours for the graphs produced

    for i, value in enumerate(per_values):
        val_data = data[value][
            sample
        ]  # Requires I index into the data when assembling plots (see plots.py)
        val_data.columns = np.arange(len(sample)) + start
        val_data = val_data.stack().reset_index(level=1)

        p = sns.scatterplot(
            data=val_data,
            x=0,
            y="level_1",
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
            0.95,
            0.95,
            value,
            horizontalalignment="right",
            verticalalignment="top",
            transform=subplots.axes_flat[i].transAxes,
            color="0.3",
        )
        p.axvline(c=palette[2], ls="--", linewidth=0.5)

        if not subplots.axes_flat[i].yaxis_inverted():
            subplots.axes_flat[i].invert_yaxis()

    if (
        label
    ):  # Chane our axis labels, here to trial number and time from the reach (which the data is aligned to!)
        to_label = subplots.to_label
        to_label.get_yaxis().get_label().set_visible(True)
        to_label.set_ylabel("Trial Number")
        to_label.get_xaxis().get_label().set_visible(True)
        to_label.set_xlabel("Time from reach (s)")

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
        legend.set_facecolor("white")

    return subplots


# Using this function we may now plot the graphs we desire!
# First let us define a function to plot spikes per unit


def per_unit_raster(data, sample=None, start=0, subplots=None, label=True):
    """
    Plots a spike raster for every unit in the given data with trials as rows.

    data is a multi-dimensional dataframe as returned from
    Experiment.align_trials(<action>, <event>, 'spike_times') indexed into to get the
    session and recording.
    """
    return _raster("unit", data, sample, start, subplots, label)


# And per trial
def per_trial_raster(data, sample=None, start=0, subplots=None, label=True):
    """
    Plots a spike raster for every trial in the given data with units as rows.

    data is a multi-dimensional dataframe as returned from
    Experiment.align_trials(<action>, <event>, 'spike_times') indexed into to get the
    session and recording.
    """
    return _raster("trial", data, sample, start, subplots, label)


# Now finally define a function that allows plotting a single unit as a raster
def single_unit_raster(data, ax=None, sample=None, start=0, unit_id=None):
    # units if plotting per trial, trials if plotting per unit
    samp_values = list(data.columns.get_level_values("trial").unique().values)
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
        y="level_1",
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
    p.axvline(c="black", ls="--", linewidth=0.5)

    if not ax.yaxis_inverted():
        ax.invert_yaxis()

    if unit_id is not None:
        p.text(
            0.95,
            0.95,
            unit_id,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            color="0.3",
        )


def _plot(level, data, rugdata, ci, subplots, label):
    values = data.columns.get_level_values(level).unique()
    values.values.sort()
    if level == "trial":
        data = data.reorder_levels(["trial", "unit"], axis=1)

    # no. units if plotting per trial, no. trials if plotting per unit
    num_samples = len(data[values[0]].columns)

    if subplots is None:
        subplots = Subplots2D(values)

    palette = sns.color_palette()

    for i, value in enumerate(values):
        val_data = data[value].stack().reset_index()
        val_data["y"] = val_data[0]
        ax = subplots.axes_flat[i]
        divider = make_axes_locatable(ax)

        r = sns.rugplot(
            ax=ax,
            palette=palette,
            data=rugdata,
            x=rugdata[0],
            height=0.1,
            legend=False,
            expand_margins=True,
        )
        r.autoscale(enable=False, tight=False)

        p = sns.lineplot(
            data=val_data,
            x="time",
            y="y",
            ci=ci,
            ax=ax,
            linewidth=0.5,
        )
        p.autoscale(enable=True, tight=False)
        p.set_yticks([])
        p.set_xticks([])
        peak = data[value].values.mean(axis=1).max()
        p.text(
            0.05,
            0.95,
            "%.1f" % peak,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            color="grey",
        )
        p.text(
            0.95,
            0.95,
            value,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            color=palette[0],
        )
        p.get_yaxis().get_label().set_visible(False)
        p.get_xaxis().get_label().set_visible(False)
        p.axvline(c=palette[1], ls="--", linewidth=0.5)
        p.set_box_aspect(1)

    if label:
        to_label = subplots.to_label
        to_label.get_yaxis().get_label().set_visible(True)
        to_label.set_ylabel("Firing rate (Hz)")
        to_label.get_xaxis().get_label().set_visible(True)
        to_label.set_xlabel("Time from push (s)")
        to_label.set_xticks(data.index)
        # to_label.set_xticklabels([- duration / 2, 0,  duration / 2])

        # plot legend subplot
        legend = subplots.legend
        legend.text(
            0,
            0.7,
            "Peak of mean",
            transform=legend.transAxes,
            color="grey",
        )
        legend.text(
            0,
            0.3,
            "Trial number" if level == "trial" else "Unit ID",
            transform=legend.transAxes,
            color=palette[0],
        )
        legend.set_visible(True)
        legend.get_xaxis().set_visible(False)
        legend.get_yaxis().set_visible(False)
        legend.set_facecolor("white")
        legend.set_box_aspect(1)

    return subplots


def per_unit_spike_rate(data, rugdata, ci=95, subplots=None, label=True):
    """
    Plots a histogram for every unit in the given session showing spike rate.

    Parameters
    ----------

    data : pandas.DataFrame
        A dataframe for a single session as returned from Behaviour.align_trials (or
        Experiment.align_trials but indexed into for session and rec_num).

    rugdata : pandas.DataFrame
        A dataframe for a single session containing the onset of the event (led_on) to plot as a rug.

    ci : int or 'sd', optional
        Confidence intervals to plot as envelope around line. Default is 95 i.e. 95%
        confidence intervals. Also accepts "'sd'" which will draw standard deviation
        envelope, or None to plot no envelope.
    """
    return _plot("unit", data, rugdata, ci, subplots, label)


def per_trial_spike_rate(data, rugdata, ci=95, subplots=None, label=True):
    """
    Plots a histogram for every trial in the given session showing spike rate.

    Parameters
    ----------

    data : pandas.DataFrame
        A dataframe for a single session as returned from Behaviour.align_trials (or
        Experiment.align_trials but indexed into for session and rec_num).

    ci : int or 'sd', optional
        Confidence intervals to plot as envelope around line. Default is 95 i.e. 95%
        confidence intervals. Also accepts "'sd'" which will draw standard deviation
        envelope, or None to plot no envelope.
    """
    return _plot("trial", data, rugdata, ci, subplots, label)


def event_times(event, myexp):
    """

    This function will give the timepoints for the specified event across experimental sessions

    event: the specific event to search for, must be input within quotes ("")

    myexp: the experiment defined in base.py

    NB: Setting the event to LED_off when the action label is correct (whose start point therefore defines LED_on as zero) will return the DURATION of the event!
    """
    times = []  # Give an empty list to store times

    for ses in myexp:
        sessiontimepoints = (
            []
        )  # Give an empty list to store this sessions times in before appending to the master list
        als = ses.get_action_labels()  # Get the action labels for this session

        for rec_num in range(
            len(als)
        ):  # This will run through all recording numbers in the above action label data
            actions = als[rec_num][:, 0]
            events = als[rec_num][:, 1]

            # Call all trials where the mouse was correct to search
            start = np.where(np.bitwise_and(actions, ActionLabels.correct))[0]

            for trial in start:
                # Now iterate through these correct trials and return all times for selected event
                event_time = np.where(
                    np.bitwise_and(
                        events[trial : trial + 10000], getattr(Events, event)
                    )
                )[0]
                sessiontimepoints.append(event_time[0])

        times.append(sessiontimepoints)

    return times


def meta_spikeglx(exp, session):
    """
    This function finds the metadata containing the depth of the channels used in the recording
    Remember must index into experiment and then files to find the spike metadata

    exp: myexp, the class defined in base.py

    session: the session to analyse, in the form of an integer (i.e., 0 for first session, 1 for second etc.)
    """

    meta = exp[session].files[0]["spike_meta"]
    data_path = myexp[session].find_file(meta)
    data = pi.read_spikeglx(data_path)

    return data
