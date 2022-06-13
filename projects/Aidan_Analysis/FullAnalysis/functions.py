"""
This is the master file containing any functions used in my analyses. 
Most require that experiments are initially defined in base.py.
sys.path.append allows the local importing of a copy of the pixels analysis repo, though this may be added to path otherwise. 

Notes
----------------------------------------------------------
|
|Some functions utilise the depreciated rec_num argument!
|
|
|
-----------------------------------------------------------
"""

from base import *  # Import our experiment and metadata needed
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from pyrsistent import b
from xml.etree.ElementInclude import include
import sys
import random
import probeinterface as pi
from reach import Session
from argparse import Action
from cv2 import bitwise_and
import matplotlib.pyplot as plt
from textwrap import wrap
from tqdm import tqdm
from channeldepth import meta_spikeglx

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
from statsmodels.graphics.factorplots import interaction_plot
from bioinfokit.analys import stat

sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")
from pixtools.utils import Subplots2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pixtools import utils
from pixtools import spike_rate
from pixels import ioutils


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


def within_unit_GLM(bin_data, myexp):

    """
    Function takes binned data (with levels in order session -> unit -> trial) and performs an ANOVA on each session
    Returns the multiple comparisons for the interaction and prints anova results
    Following parameters are used:

    IV = Firing Rate, DV = Bin, Unit (containing many trials)

    Model is as follows:

    firing_rate ~ C(bin) + C(unit) + C(bin):C(unit)

    bin_data: the output of the per_trial_binning function reordered to swap unit and trial level hierarchy

    myexp: experimental cohort defined in base.py


    NB: This function runs the model as a distributed set of calculations!!
    i.e., it slices the data and runs several smaller calculations before concatenating results

    """
    # store data as a resillient distributed dataset
    # will do this using pyspark

    multicomp_bin_results = []
    multicomp_unit_results = []
    multicomp_int_results = []

    # Now split this data into sessions
    for s, session in tqdm(enumerate(myexp)):

        sessions = bin_data.columns.get_level_values(0).unique()
        name = session.name
        ses_data = bin_data[sessions[s]]

        # Now shall average the data per bin
        ses_avgs = bin_data_average(ses_data)

        # Then melt data to prep for analysis
        ses_avgs = ses_avgs.reset_index().melt(id_vars=["unit", "trial"])
        ses_avgs.rename({"value": "firing_rate"}, axis=1, inplace=True)

        ##Now shall construct a linear model to allow for analysis
        # IV: Firing Rate
        # DVs: Bin (within subjects), Unit (Within subjects - trials averaged out)

        # Now run the GLM analysis
        model = ols(
            "firing_rate ~ C(bin) + C(unit) + C(bin):C(unit)", data=ses_avgs
        ).fit()

        output = sm.stats.anova_lm(model, typ=2)  # Type II SS as this is faster
        print(f"ANOVA Results - Session {name}")
        print(output)

        ####Run Multiple Comparisons####
        # Main effect of bin
        res = stat()

        print(f"starting multiple comparisons for session {name}")
        # Interaction
        res.tukey_hsd(
            df=ses_avgs,
            res_var="firing_rate",
            xfac_var=["bin", "unit"],
            anova_model="firing_rate ~ C(bin) + C(unit) + C(bin):C(unit)",
            phalpha=0.05,
            ss_typ=2,
        )
        int_effect = res.tukey_summary
        multicomp_int_results.append(int_effect)
    return multicomp_int_results


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


def significance_extraction(CI):
    """
    This function takes the output of the get_aligned_spike_rate_CI function and extracts any significant values, returning a dataframe in the same format.

    CI: The dataframe created by the CI calculation previously mentioned

    """

    sig = []
    keys = []
    rec_num = 0

    # This loop iterates through each column, storing the data as un, and the location as s
    for s, unit in CI.items():
        # Now iterate through each recording, and unit
        # Take any significant values and append them to lists.
        if unit.loc[2.5] > 0 or unit.loc[97.5] < 0:
            sig.append(
                unit
            )  # Append the percentile information for this column to a list
            keys.append(
                s
            )  # append the information containing the point at which the iteration currently stands

    # Now convert this list to a dataframe, using the information stored in the keys list to index it
    sigs = pd.concat(
        sig, axis=1, copy=False, keys=keys, names=["session", "unit", "rec_num"]
    )

    return sigs


def percentile_plot(CIs, sig_CIs, exp, sig_only=False, dir_ascending=False):
    """

    This function takes the CI data and significant values and plots them relative to zero.
    May specify if percentiles should be plotted in ascending or descending order.

    CIs: The output of the get_aligned_spike_rate_CI function, i.e., bootstrapped confidence intervals for spike rates relative to two points.

    sig_CIs: The output of the significance_extraction function, i.e., the units from the bootstrapping analysis whose confidence intervals do not straddle zero

    exp: The experimental session to analyse, defined in base.py

    sig_only: Whether to plot only the significant values obtained from the bootstrapping analysis (True/False)

    dir_ascending: Whether to plot the values in ascending order (True/False)

    """
    # First sort the data into long form for both datasets, by percentile
    CIs_long = (
        CIs.reset_index()
        .melt("percentile")
        .sort_values("value", ascending=dir_ascending)
    )
    CIs_long = CIs_long.reset_index()
    CIs_long["index"] = pd.Series(
        range(0, CIs_long.shape[0])
    )  # reset the index column to allow ordered plotting

    CIs_long_sig = (
        sig_CIs.reset_index()
        .melt("percentile")
        .sort_values("value", ascending=dir_ascending)
    )
    CIs_long_sig = CIs_long_sig.reset_index()
    CIs_long_sig["index"] = pd.Series(range(0, CIs_long_sig.shape[0]))

    # Now select if we want only significant values plotted, else raise an error.
    if sig_only is True:
        data = CIs_long_sig

    elif sig_only is False:
        data = CIs_long

    else:
        raise TypeError("Sig_only argument must be a boolean operator (True/False)")

    # Plot this data for the experimental sessions as a pointplot.
    for s, session in enumerate(exp):
        name = session.name

        p = sns.pointplot(
            x="unit",
            y="value",
            data=data.loc[(data.session == s)],
            order=data.loc[(data.session == s)]["unit"].unique(),
            join=False,
            legend=None,
        )  # Plots in the order of the units as previously set, uses unique values to prevent double plotting

        p.set_xlabel("Unit")
        p.set_ylabel("Confidence Interval")
        p.set(xticklabels=[])
        p.axhline(0)
        plt.suptitle(
            "\n".join(
                wrap(
                    f"Confidence Intervals By Unit - Grasp vs. Baseline - Session {name}"
                )
            )
        )  # Wraps the title of the plot to fit on the page.

        plt.show()


def per_trial_binning(trial, myexp, timespan, bin_size):
    """
    Function takes data aligned by trial (i.e., using myexp align_trials method),
    bins them, then returns the dataframe with an added bin index

    The end result of this is an index structured as (time, 'lower bin, upper bin')

    trial: The data aligned to an event according to myexp.align_trials

    myexp: The experimental cohort defined in base.py

    timespan: the total time data was aligned to in myexp.align_trials (i.e., duration) in seconds

    bin size: the size of the bins in seconds

    """

    # First reorder data hierarchy
    reorder = trial.reorder_levels(["session", "trial", "unit"], axis=1)

    # Perform binning on  whole dataset
    # First calculate the number of bins to create based on total duration and bin length
    bin_number = int(timespan / bin_size)

    values = reorder.columns.get_level_values(
        "trial"
    ).unique()  # Get trials for session

    # Add a new level containing the bin data for the dataframe
    bin_vals = pd.cut(reorder.index, bin_number, include_lowest=True, precision=1)
    bin_vals = bin_vals.astype(str)  # Convert the interval series to a list of strings

    # Remove brackets from bin_vals
    bin_vals_str = []
    for t in bin_vals:
        t = t.strip("()[]")
        bin_vals_str.append(t)

    # This will be added as an index, before time.
    full_data = reorder.assign(
        bin=bin_vals_str
    )  # Add bin values temporarily as a column
    full_data.set_index(
        "bin", append=True, inplace=True
    )  # Convert this column to a new index, below time
    return full_data


def bin_data_average(bin_data):
    """
    This function takes the dataframe produced by per_trial_binning() including bins as a multiindex
    and takes the average of each bin's activity within a given time period
    This returns the average firing rate of each unit (i.e., per session and time)

    bin_data: the dataframe produced by per_trial_binning, with predefined bins

    NB: To return a transposed version of the data (i.e., with bin averages as rows rather than columns)
        add ".T" to function call.
    """

    # First extract the values of each bin in bin_data
    bins = bin_data.index.get_level_values(
        "bin"
    )  # a list of the bins created by the previous function
    bin_names = (
        bins.unique()
    )  # Take the unique values present, giving us the list of bin categories

    # Take averages for each bin, across all units and trials then concatenate into a dataframe, where each column is a bin
    binned_avgs = pd.concat(
        (bin_data.loc[bins == b].mean(axis=0) for b in bin_names),
        axis=1,
        keys=bin_names,
    )
    binned_avgs = pd.DataFrame(binned_avgs)

    # Return the data
    return binned_avgs


def bin_data_bootstrap(
    bin_data, CI=95, ss=20, bootstrap=10000, name=None, cache_overwrite=False
):
    """
    This function takes the data, with bin averages calculated by bin_data_average and calculates boostrapped confidence intervals
    To do so, it takes a random sample many times over and uses this data to calculate percentiles

    bin_data: the dataframe produced by bin_data_average, with bin times represented as columns

    CI: the confidence interval to calculate (default is 95%)

    ss: the sample size to randomly select from the averages (default is 20)

    bootstrap: the number of times a random sample will be taken to create a population dataset (default is 10,000)

    name: the name to save the output under in cache, speeds the process if this has been run before

    cache_overwrite: whether to overwrite the saved cache with new data, (default is False)
    """

    ##Stages of this analysis are as follows:
    # 1. Take individual bin
    # 2. Calculate confidence interval via bootstapping
    # 3. Store all bin CIs

    # Check if the function has been run before, and saved to a chache
    # NB: This function caches under the name of the first session in the list!
    if name is not None:
        # Read in data from previous run
        output = myexp.sessions[0].interim / "cache" / (name + ".h5")
        if output.exists() and cache_overwrite is not True:
            df = ioutils.read_hdf5(myexp.sessions[0].interim / "cache" / (name + ".h5"))
            return df

    # First define confidence intervals
    lower = (100 - CI) / 2  # Lower confidence interval bound (default 2.5%)
    upper = 100 - lower  # Upper bound (default 97.5%)
    percentiles = [lower, 50, upper]  # The defined percentiles to analyse
    cis = []  # empty list to store calculated intervals

    bin_values = bin_data.columns
    keys = []

    # Loop through each of the bin averages and calculate their confidence interval
    for s, session in enumerate(myexp):
        names = session.name
        ses_bins = bin_data[bin_data.index.get_level_values("session") == s]
        trials = ses_bins.index.get_level_values(
            "trial"
        ).unique()  # get the trials for a given session

        for t in trials:
            trial_bins = ses_bins[ses_bins.index.get_level_values("trial") == t]

            # Take every bin within a trial
            for b in bin_values:
                bin_avg = trial_bins[b]
                samples = np.array(
                    [
                        [np.random.choice(bin_avg.values, size=ss)]
                        for j in range(bootstrap)
                    ]
                )

                # Use these samples to calculate the percentiles for each bin
                medians = np.median(samples, axis=2)  # Median of the samples taken
                results = np.percentile(
                    medians, percentiles, axis=0
                )  # The percentiles, relative to the median
                cis.append(pd.DataFrame(results))
                keys.append(
                    (s, t, b)
                )  # Set the keys for the dataframe to be bin, session, trial

    # Now save these calculated confidence intervals to a dataframe
    df = pd.concat(cis, axis=1, names=["session", "trial", "bin"], keys=keys)
    df.set_index(
        pd.Index(percentiles, name="percentile"), inplace=True
    )  # Set the name of the index to percentiles
    df.columns = df.columns.droplevel(
        level=3
    )  # remove the redundant units row, as they have been averaged out

    # Finally, save this df to cache to speed up future runs
    if name is not None:
        ioutils.write_hdf5(myexp.sessions[0].interim / "cache" / (name + ".h5"), df)
        return df

    return df


def bs_pertrial_bincomp(bs_data, myexp):
    """
    This function imports percentile data produced from bootstrapping, and compares across bins to determine if a significant change from previous states occurs
    This process is repeated across trials to allow plotting of significance.
    Returns a dataframe with bins classified as either increasing, decreasing or displaying little change from previous timepoint.

    bs_data: the percentile information produced by bin_data_bootstrap()

    myexp: the experimental cohort defined in base.py
    """

    total_bin_response = {}
    # First import the data and split it by session, and trial.
    for s, session in enumerate(myexp):
        name = session.name
        ses_data = bs_data[0]
        ses_bin_response = {}
        for t, cols in enumerate(ses_data):
            trial_data = ses_data[
                cols[0]
            ]  # All bins, and associated percentiles for a given trial

            # Now that we have all the bins for a given trial,
            # t count value in enumerate loop will allow comparison to previous iteration
            bins = trial_data.columns

            # compare each column's 2.5% (lower) to previous column's 97.5%
            # to check for sig. increase in activity
            previous_bin = []
            trial_bin_response = {}  # store as a dictionary
            for b, bins in enumerate(trial_data):

                # First store the initial bin to previous_bin
                if len(previous_bin) == 0:
                    previous_bin.append(trial_data[bins])
                    continue

                # Once this has been done, will compare across bins
                current_bin = trial_data[bins]

                # First check if there is a sig. increase - i.e., if current 2.5 and previous 97.5 do not overlap
                if current_bin[2.5] >= previous_bin[0][97.5]:
                    trial_bin_response.update({bins: "increased"})

                # Now check if there is a sig. decrease
                elif current_bin[97.5] <= previous_bin[0][2.5]:
                    trial_bin_response.update({bins: "decreased"})

                # And finally if there is no change at all
                else:
                    trial_bin_response.update({bins: "nochange"})

                # The replace previous bin with current bin
                previous_bin = []
                previous_bin.append(current_bin)

            # Now append data for each trial to a larget list
            ses_bin_response.update(
                {cols[0]: trial_bin_response}
            )  # contains every trial for a given session
        total_bin_response.update({name: ses_bin_response})

    # Convert nested dictionary to dataframe
    df = pd.DataFrame.from_dict(
        {
            (i, j): total_bin_response[i][j]
            for i in total_bin_response.keys()
            for j in total_bin_response[i].keys()
        },
        orient="index",
    )
    df.index.rename(["session", "trial"], inplace=True)
    df.columns.names = ["bin"]
    return df


def bs_graph(aligned_data, binned_data, bin_comparisons, myexp):
    """
    Function plots the binned data graph including period of LED onset for each trial within a given session.

    aligned_data: the dataset produced by align_trials

    binned_data: the dataset produced by per_trial_binning

    bin_comparisons: the dataset containing comparisons between bins, produced by bs_pertrial_bincomp

    myexp: the experimental cohort defined in base.py

    """
    for s, session in enumerate(myexp):
        name = session.name

        ses_data = binned_data[s]
        ses_values = ses_data.columns.get_level_values(
            "trial"
        ).unique()  # Get all trial values for the session
        ses_start_times = start[
            name
        ]  # LED on time for the session, will leave trial numbers in ses_values unsorted to ensure they align

        ses_bin_comps = bin_comparisons[
            bin_comparisons.index.get_level_values("session") == name
        ]

        num_trials = len(
            ses_data[ses_values[0]].columns
        )  # How many trials shall be plotted
        subplots = Subplots2D(
            ses_values, sharex=True, sharey=True
        )  # determines the number of subplots to create (as many as there are trials in the session)

        for i, value in enumerate(ses_values):
            data = ses_data[value].stack().reset_index()
            data["y"] = data[0]  # Add a new column containing y axis values
            ax = subplots.axes_flat[i]  # Plot this on one graph

            # bin responses for the trial to be plotted
            trial_bins = ses_bin_comps[
                ses_bin_comps.index.get_level_values("trial") == i
            ]

            # Now create the actual lineplot
            p = sns.lineplot(x="time", y="y", data=data, ax=ax, ci="sd")

            # Now for each of these plots, add the LED on-off time as a shaded area
            p.axvspan(ses_start_times[i], 0, color="green", alpha=0.3)

            ##Now, within each trial, add shaded areas representing the binned timescales
            for i, cols in enumerate(ses_bin_comps.columns):
                shaded_bin = data["time"].loc[data["bin"] == ses_bin_comps.columns[i]]
                bin_significance = trial_bins[cols]

                # Then plot an indicator of significance to these bins
                if bin_significance.values == "nochange":
                    p.axvspan(
                        shaded_bin.values[0],
                        shaded_bin.values[-1],
                        color="gray",
                        alpha=0,
                        hatch=r"//",
                    )  # Change this alpha value to above zero to plot nonsignificant bins.

                if (
                    bin_significance.values == "increased"
                    or bin_significance.values == "decreased"
                ):
                    p.axvspan(
                        shaded_bin.values[0],
                        shaded_bin.values[-1],
                        color="red",
                        alpha=0.1,
                        hatch=r"/o",
                    )

            # Now remove axis labels
            p.get_yaxis().get_label().set_visible(False)
            p.get_xaxis().get_label().set_visible(False)

            # Then set the x limit to be one second post trial completion
            p.set(xlim=(-5, 1))

        # Now plot the legend for each of these figures
        legend = subplots.legend

        legend.text(0.3, 0.5, "Trial", transform=legend.transAxes)
        legend.text(
            -0.2, 0, "Firing Rate (Hz)", transform=legend.transAxes, rotation=90
        )  # y axis label
        legend.text(
            0, -0.3, "Time Relative to LED Off (s)", transform=legend.transAxes
        )  # x axis label

        legend.set_visible(True)
        legend.get_xaxis().set_visible(False)
        legend.get_yaxis().set_visible(False)
        legend.set_facecolor("white")
        legend.set_box_aspect(1)

        # Display the plot
        plt.gcf().set_size_inches(20, 20)

        # Uncomment to save figures as pdf.
        utils.save(
            f"/home/s1735718/Figures/{myexp[s].name}_spikerate_LED_Range_PerTrial",
            nosize=True,
        )


def cross_trial_bootstrap(
    bin_data, CI=95, ss=20, bootstrap=10000, cache_name=None, cache_overwrite=False
):
    """
    This function takes the averages across trials for a given bin as previously calculated
    and samples across values, to give a series of percentiles for average firing rate in a given bin

    bin_data: the dataframe produced by bin_data_average, with bin times represented as columns

    CI: the confidence interval to calculate (default is 95%)

    ss: the sample size to randomly select from the averages (default is 20)

    bootstrap: the number of times a random sample will be taken to create a population dataset (default is 10,000)

    cache_name: the name to save the output under in cache, speeds the process if this has been run before

    cache_overwrite: whether to overwrite the saved cache with new data, (default is False)
    """

    ##Stages of this analysis are as follows:
    # 1. Take individual bin
    # 2. Calculate confidence interval via bootstapping
    # 3. Store all bin CIs

    # Check if the function has been run before, and saved to a chache
    # NB: This function caches under the name of the first session in the list!

    if cache_name is not None:
        # Read in data from previous run
        output = myexp.sessions[0].interim / "cache" / (name + ".h5")
        if output.exists() and cache_overwrite is not True:
            df = ioutils.read_hdf5(myexp.sessions[0].interim / "cache" / (name + ".h5"))
            return df

    # First define confidence intervals
    lower = (100 - CI) / 2  # Lower confidence interval bound (default 2.5%)
    upper = 100 - lower  # Upper bound (default 97.5%)
    percentiles = [lower, 50, upper]  # The defined percentiles to analyse
    cis = []  # empty list to store calculated intervals

    bin_values = bin_data.columns
    keys = []

    # Loop through each of the bin averages and calculate their confidence interval
    for s, session in enumerate(myexp):

        names = session.name
        ses_bins = bin_data[bin_data.index.get_level_values("session") == names]
        trials = ses_bins.index.get_level_values(
            "trial"
        ).unique()  # get the trials for a given session

        # Sample from every trial within a bin
        for b in bin_values:

            # Take every trial within a bin
            bin_avgs = ses_bins[b]
            samples = np.array(
                [[np.random.choice(bin_avgs.values, size=ss)] for j in range(bootstrap)]
            )

            # Use these samples to calculate the percentiles for each bin
            medians = np.median(samples, axis=2)  # Median of the samples taken
            results = np.percentile(
                medians, percentiles, axis=0
            )  # The percentiles, relative to the median
            cis.append(pd.DataFrame(results))
            keys.append(
                (names, b)
            )  # Set the keys for the dataframe to be bin, session, trial

    # Now save these calculated confidence intervals to a dataframe
    df = pd.concat(cis, axis=1, names=["session", "bin"], keys=keys)
    df.set_index(
        pd.Index(percentiles, name="percentile"), inplace=True
    )  # Set the name of the index to percentiles
    df.columns = df.columns.droplevel(
        level=2
    )  # remove the redundant units row, as they have been averaged out

    # Finally, save this df to cache to speed up future runs
    if cache_name is not None:
        ioutils.write_hdf5(
            myexp.sessions[0].interim / "cache" / (cache_name + ".h5"), df
        )
        return df

    return df


def cross_trial_bincomp(bs_data, myexp):
    """
    This function imports percentile data produced from bootstrapping, and compares across bins to determine if a significant change from previous states occurs
    This process is repeated across trials to allow plotting of significance.
    Returns a dataframe with bins classified as either increasing, decreasing or displaying little change from previous timepoint.

    bs_data: the percentile information produced by bin_data_bootstrap()

    myexp: the experimental cohort defined in base.py
    """

    total_bin_response = {}
    # First import the data and split it by session, and trial.
    for s, session in enumerate(myexp):
        name = session.name
        ses_data = bs_data[name]
        ses_bin_response = {}
        for t, cols in enumerate(ses_data):

            # Now that we have all the bins for a given trial,
            # t count value in enumerate loop will allow comparison to previous iteration
            bins = ses_data.columns

            # compare each column's 2.5% (lower) to previous column's 97.5%
            # to check for sig. increase in activity
            previous_bin = []
            for b, bins in enumerate(ses_data):

                # First store the initial bin to previous_bin
                if len(previous_bin) == 0:
                    previous_bin.append(ses_data[bins])
                    continue

                # Once this has been done, will compare across bins
                current_bin = ses_data[bins]

                # First check if there is a sig. increase - i.e., if current 2.5 and previous 97.5 do not overlap
                if current_bin[2.5] >= previous_bin[0][97.5]:
                    ses_bin_response.update({bins: "increased"})

                # Now check if there is a sig. decrease
                elif current_bin[97.5] <= previous_bin[0][2.5]:
                    ses_bin_response.update({bins: "decreased"})

                # And finally if there is no change at all
                else:
                    ses_bin_response.update({bins: "nochange"})

                # The replace previous bin with current bin
                previous_bin = []
                previous_bin.append(current_bin)

            # Now append data for each trial to a larget list
        total_bin_response.update({name: ses_bin_response})

    # Convert nested dictionary to dataframe
    df = pd.DataFrame.from_dict(
        total_bin_response,
        orient="index",
    )
    # df.index.rename(["session"], inplace=True)
    df.columns.names = ["bin"]
    df.index.rename("session", inplace=True)
    return df


def percentile_range(bs_data):
    """
    Function takes percentiles for given bins and calculates the range, adding this as a new index row

    bs_data: a bootstrapped dataset containing 2.5, 50, and 97.5% percentiles as index
             with data organised as columns for some given bin
    """

    # Run through the set col by col
    ranges = []

    for col in bs_data:
        column = bs_data[col]
        perc_delta = column[97.5] - column[2.5]  # Range of the percentiles
        ranges.append(perc_delta)

    # Now append this as a new row
    bs_data.loc["range"] = ranges
    return bs_data


def variance_boostrapping(
    bs_data,
    myexp,
    CI=95,
    ss=20,
    bootstrap=10000,
    cache_name=None,
    cache_overwrite=False,
):
    """
    Similar to previous functions, this performs a boostrapping analysis per bin throughout a sessions trials.
    However, it instead analyses if the size of the variance (ie. the range between percentiles calculated previously) differs significantly between bins
    This will act as a proxy for the synchronicity of neuronal activity for a given bin.

    bs_data: the data calculated within each trial by bin_data_bootstrap (NOT Cross data bootstrap!)

    myexp: the experimental cohort defined in base.py

    CI: the confidence interval to calculate (default is 95%)

    ss: the sample size to randomly select from the averages (default is 20)

    bootstrap: the number of times a random sample will be taken to create a population dataset (default is 10,000)

    cache_name: the name to save the output under in cache, speeds the process if this has been run before

    cache_overwrite: whether to overwrite the saved cache with new data, (default is False)

    """
    # Check if the function has been run before, and saved to a chache
    # NB: This function caches under the name of the first session in the list!
    if cache_name is not None:
        # Read in data from previous run
        output = myexp.sessions[0].interim / "cache" / (cache_name + ".h5")
        if output.exists() and cache_overwrite is not True:
            df = ioutils.read_hdf5(
                myexp.sessions[0].interim / "cache" / (cache_name + ".h5")
            )
            return df

    # First define confidence intervals
    lower = (100 - CI) / 2  # Lower confidence interval bound (default 2.5%)
    upper = 100 - lower  # Upper bound (default 97.5%)
    percentiles = [lower, 50, upper]  # The defined percentiles to analyse
    cis = []  # empty list to store calculated intervals

    keys = []

    for s, session in enumerate(myexp):
        ses_data = bs_data[s]
        ses_data = ses_data.loc[ses_data.index == "range"]
        ses_data = ses_data.melt()

        name = session.name
        legend_names.append(name)
        bin_vals = ses_data.bin.unique()

        # Sample across all trial ranges for a given bin
        for b in bin_vals:
            bin_data = ses_data.loc[ses_data.bin == b]
            samples = np.array(
                [[np.random.choice(bin_data.value, size=ss)] for j in range(bootstrap)]
            )

            # Save results of single bin boostrapping
            medians = np.median(samples, axis=2)  # Median of the samples taken
            results = np.percentile(
                medians, percentiles, axis=0
            )  # The percentiles, relative to the median
            cis.append(pd.DataFrame(results))

            keys.append((name, b))

    # Now save these calculated confidence intervals to a dataframe
    df = pd.concat(cis, axis=1, names=["session", "bin"], keys=keys)
    df.set_index(
        pd.Index(percentiles, name="percentile"), inplace=True
    )  # Set the name of the index to percentiles
    df.columns = df.columns.droplevel(
        level=2
    )  # remove the redundant units row, as they have been averaged out

    if cache_name is not None:
        ioutils.write_hdf5(
            myexp.sessions[0].interim / "cache" / (cache_name + ".h5"), df
        )
        return df

    return df


def bin_data_SD(bin_data):
    """
    Function is similar to bin_data_mean.

    This function takes the dataframe produced by per_trial_binning() including bins as a multiindex
    and calculates the standard deviation of each bin's activity within a given time period
    This returns the SD of the firing rate of each unit (i.e., per session and time)

    bin_data: the dataframe produced by per_trial_binning, with predefined bins

    NB: To return a transposed version of the data (i.e., with bin averages as rows rather than columns)
        add ".T" to function call.
    """

    # First extract the values of each bin in bin_data
    bins = bin_data.index.get_level_values(
        "bin"
    )  # a list of the bins created by the previous function
    bin_names = (
        bins.unique()
    )  # Take the unique values present, giving us the list of bin categories
    ses_bins = {}
    for s, session in enumerate(myexp):
        name = session.name

        ses_data = bin_data[s]
        trial_bins = {}

        # Trial number will give the following loop the number of times to iterate
        trial_number = ses_data.columns.get_level_values("trial").unique()
        # Now, take each trial and average across bins
        for t in trial_number:
            trial_data = ses_data[t]
            individual_bins = {}
            for b in bin_names:

                bin_vals = trial_data.loc[trial_data.index.get_level_values("bin") == b]
                bin_st_dev = np.std(
                    bin_vals.values
                )  # take standard deviation for all values in this trials bin
                individual_bins.update({b: bin_st_dev})
            trial_bins.update({t: individual_bins})

        ses_bins.update({name: trial_bins})

    # Merge these SDs as a dataframe
    binned_sd = pd.DataFrame.from_dict(
        {(i, j): ses_bins[i][j] for i in ses_bins.keys() for j in ses_bins[i].keys()},
        orient="index",
    )

    # Take the mean of the SDs for each trial Bin

    binned_sd.index.rename(["session", "trial"], inplace=True)
    binned_sd.columns.names = ["bin"]

    # Return the data
    return binned_sd


def unit_depths(exp):
    """
    Parameters
    ==========
    exp : pixels.Experiment
        Your experiment.

    This is an updated version of the pixtools function
    """
    info = exp.get_cluster_info()
    depths = []
    keys = []

    for s, session in enumerate(exp):
        name = session.name
        session_depths = {}
        rec_num = 0
        for rec_num, probe_depth in enumerate(session.get_probe_depth()):
            rec_depths = {}
            rec_info = info[s]
            id_key = "id" if "id" in rec_info else "cluster_id"  # Depends on KS version

            for unit in rec_info[id_key]:
                unit_info = rec_info.loc[rec_info[id_key] == unit].iloc[0].to_dict()
                rec_depths[unit] = probe_depth - unit_info["depth"]

            session_depths = pd.DataFrame(rec_depths, index=["depth"])
        keys.append(name)
        session_depths = pd.concat([session_depths], axis=1, names="unit")
        depths.append(session_depths)

    return pd.concat(depths, axis=1, names=["session", "unit"], keys=keys)


def unit_delta(glm, myexp, bin_data, bin_duration, sig_only, percentage_change):
    """
    Function takes the output of the multiple comparisons calculated by within_unit_GLM() and calculates the larges change in firing rate (or other IV measured by the GLM) between bins, per unit
    This requires both experimental cohort data, and bin_data calculated by per_trial_binning() and passed through reorder levels (as "session", "unit", "trial")
    Function returns a list of lists, in a hierarchy of [session[unit deltas]]

    Using this data, a depth plot may be created displaying the relative delta by unit location in pM2

    glm: the result of the ANOVA and subsequent Tukey adjusted multiple comparisons performed by within_unit_GLM
         NB: THIS IS A COMPUTATIONALLY EXPENSIVE FUNCTION, DO NOT RUN IT ON MORE THAN ONE SESSION AT A TIME

    myexp: the experimental cohort defined in base.py

    bin_data: the binned raw data computed by the per_trial_binning function

    bin_duration: the length of the bins in the data

    sig_only: whether to return only the greatest delta of significant bins or not

    percentage_change: whether to return deltas as a percentage change

    """
    ses_deltas = []
    for s, session in enumerate(myexp):

        final_deltas = []
        unit_deltas = []
        sigunit_comps = []
        ses_comps = glm[s]

        if sig_only is True:
            # Return only significant comparisons
            ses_comps = ses_comps.loc[ses_comps["p-value"] < 0.05]

        # Determine list of units remaining in comparison
        units = []
        for i in ses_comps["group1"]:
            units.append(i[1])
        units = np.array(units)
        units = np.unique(units)

        # Now iterate through comparisons, by unit, Saving the significant comparisons to allow later calculation of delta

        for i in tqdm(units):
            unit_comps = ses_comps[
                ses_comps["group1"].apply(lambda x: True if i in x else False)
            ]  # Lambda function checks if the value is in the tuple for group one
            unit_comps = unit_comps[
                unit_comps["group2"].apply(lambda x: True if i in x else False)
            ]
            unit_comps.reset_index(inplace=True)

            # find row with largest difference
            if unit_comps.empty:  # skip if empty
                continue

            unit_comps = unit_comps.sort_values("Diff", ascending=False)
            ##Right around here I need to work out a way to sort by adjacent bins only
            # Extract only adjacent bins, then take the largest value with this set.
            for item in unit_comps.iterrows():
                # row = unit_comps["Diff"].idxmax()
                row = item[1]
                g1_bin, unit_num = item[1]["group1"]
                g1_bin1 = round(float(g1_bin.split()[0][0:-1]), 1)
                g1_bin2 = round(float(g1_bin.split()[1][0:]), 1)

                g2_bin, unit_num = item[1]["group2"]
                g2_bin1 = round(float(g2_bin.split()[0][0:-1]), 1)
                g2_bin2 = round(float(g2_bin.split()[1][0:]), 1)

                # Check if bins are sequential, will return the largest value where bins are next to eachother
                if g2_bin1 == g1_bin2 + bin_duration:

                    sigunit_comps.append([row])
                else:
                    continue

        # Now that we know the units with significant comparisons, take these bins from raw firing rate averages
        ses_avgs = bin_data_average(bin_data[s])
        # Iterate through our significant comparisons, calculating the actual delta firing rate
        for i in range(len(sigunit_comps)):
            sig_comp = sigunit_comps[i]
            sig_comp = pd.DataFrame(sig_comp)  # convert list to dataframe

            unit = [x[1] for x in sig_comp["group1"]]
            ses_unit = ses_avgs.loc[
                ses_avgs.index.get_level_values("unit") == int(unit[0])
            ].mean()  # get all rows where our units firing rate is present

            # Get the bins that the signficant comparison looked at
            bin_val1 = [x[0] for x in sig_comp["group1"]][0]
            bin_val2 = [x[0] for x in sig_comp["group2"]][0]

            # Return the percentage change from initial state rather than raw value
            if percentage_change == True:
                change = ses_unit[bin_val2] - ses_unit[bin_val1]
                delta = (change / ses_unit[bin_val1]) * 100
                if ses_unit[bin_val1] == 0:
                    continue  # Skip this value, if it changes from a value of zero, this is infinite
                unit_deltas.append([int(unit[0]), delta])
            # Finally, get the delta value across these bins for the given unit
            elif percentage_change == False:
                delta = ses_unit[bin_val2] - ses_unit[bin_val1]
                unit_deltas.append([int(unit[0]), delta])

        # Iterate through unit_deltas and remove any duplicate units
        # Keeping only the units of largest values
        unit_deltas = pd.DataFrame(unit_deltas, columns=["unit", "delta"])
        for j in unit_deltas["unit"].unique():
            vals = unit_deltas.loc[unit_deltas["unit"] == j]
            vals = vals.sort_values("delta", key=abs, ascending=False)

            final_deltas.append(vals.iloc[0].values.tolist())

        ses_deltas.append(final_deltas)

    return ses_deltas
