#%%
# First shall import required packages

import enum
from xml.etree.ElementInclude import include
from matplotlib.pyplot import title, ylabel, ylim
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from base import *


print("starting function import")
# Will import functions from the master file to speed this process and avoid un-needed bootstrapping
from functions import (
    per_trial_binning,
    bin_data_average,
    bin_data_SD,
    bin_data_bootstrap,
    bs_pertrial_bincomp,
)

print("import function complete")
from CI_Analysis import significance_extraction
from CI_Analysis import percentile_plot
from functions import event_times, per_trial_spike_rate
from textwrap import wrap

sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")
sys.path.insert(0, "/home/s1735718/PixelsAnalysis/pixels/pixels")
from pixtools.utils import Subplots2D  # use the local copy of base.py
from pixtools import utils
from pixtools import spike_rate
from pixels import ioutils


# Now select the units that shall be analysed
# Will only analyse M2 for the purposes of this script
units = myexp.select_units(group="good", name="m2", min_depth=200, max_depth=1200)

# Then align trials to LED off (i.e, end of trial) with the time window determined by the max trial length
duration = 10
trial_aligned = myexp.align_trials(
    ActionLabels.correct, Events.led_off, "spike_rate", duration=duration, units=units
)

#%%
##################################################################################################
##Run functions defined in unit_binning_analysis to perform the binning and boostrapping of data##
##################################################################################################

print("bootstrapping initiated")

bin_data = per_trial_binning(trial_aligned, myexp, timespan=10, bin_size=0.1)

bin_avgs = bin_data_average(bin_data)

bootstrap = bin_data_bootstrap(
    bin_avgs, CI=95, ss=20, bootstrap=10000, name="fullbootstrap", cache_overwrite=True
)

# bin_comp = bs_pertrial_bincomp(bootstrap, myexp)

#%%
##################################################################################################################
# As an alternative to the bootstrapping approach taken before I will now take average firing rates across trials
# All trials will be combined into a single dataset and binned
# Each bin will contain a series of average firing rates across trials
# May then bootstrap this dataset to see how firing rate changes with time
##################################################################################################################

# First, average firing rate for each session across units/trials (to give a vals for each trial per bin)
# i.e., each bin will have a single value representing the trials average firing rate at that point
# To do this, will average across units, in bin_avgs

bins = bin_avgs.columns
ses_bins = {}

for s, session in enumerate(myexp):
    name = session.name

    ses_data = bin_avgs.loc[bin_avgs.index.get_level_values("session") == s]
    ses_trials = ses_data.index.get_level_values("trial")
    trial_bins = {}
    # Now, take each trial and average across bins
    for t in ses_trials:
        trial_data = ses_data.loc[ses_data.index.get_level_values("trial") == t].mean(
            axis=0
        )
        trial_bins.update({t: trial_data})

    ses_bins.update({name: trial_bins})

# Merge these averages as a dataframe
means = pd.DataFrame.from_dict(
    {(i, j): ses_bins[i][j] for i in ses_bins.keys() for j in ses_bins[i].keys()},
    orient="index",
)

means.index.rename(["session", "trial"], inplace=True)
means.columns.names = ["bin"]

# %%
# Define all functions to be used in this script
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


#%%
# Now that the firing rates have been averaged, may perform a bootstrapping analysis per bin

crosstrialbs = cross_trial_bootstrap(
    means,
    CI=95,
    ss=20,
    bootstrap=10000,
    cache_name="crosstrialbs",
    cache_overwrite=False,
)

#%%
# Using this boostrapped firing rate, will plot average firing rate across trials, then compare bootstrap CIs
# Specifically, compare size of the variance.

# cross_trial_variance_change = percentile_range(
#     crosstrialbs
# )  # How the variance changes with time averaged across all trials for a session
# within_trial_variance_change = percentile_range(
#     bootstrap
# )  # How the variance changes with time within each trial

sd_change = bin_data_SD(bin_data)


# Plot this change in variance to see how it changes with time, relative to LED on
# First, for every trial in a session averaged, how does the variance change with time
legend_names = []
plt.rcParams.update({"font.size": 12})
for s, session in enumerate(myexp):

    name = session.name
    legend_names.append(name)
    ses_data = sd_change.loc[sd_change.index.get_level_values("session") == name]
    ses_means = ses_data.mean(axis=0)

    bin_duration = duration / len(sd_change.columns)
    p = sns.lineplot(data=ses_means, palette="hls")

    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are on
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off

# Plot signficance annotations
x1 = "-0.6, -0.5"  # first col
x2 = "-0.1, 0.0005"  # last col
p.axvspan(
    x1,
    x2,
    color="red",
    alpha=0.1,
)

# Adjust labels/legend
plt.ylabel("Average Variance in Firing Rate (SD)")
plt.xlabel(
    f"Trial Duration Elapsed Relative to LED Off (as {bin_duration}s Bins)", labelpad=20
)
plt.legend(
    labels=legend_names,
    bbox_to_anchor=(1.1, 1.1),
    title="Session (RecordingDate_MouseID)",
)
plt.suptitle("Average Trial Firing Rate Variance")
plt.axvline(
    x=round((len(ses_data.columns) / 2)), ls="--", color="green"
)  # Plot vertical line at midpoint, representing the event the data is aligned to

utils.save(
    f"/home/s1735718/Figures/AllSession_Average_Variance_Change",
    nosize=True,
)
plt.show()

#%%
# Now plot average firing rate relative to led off to go with this

legend_names = []
for s, session in enumerate(myexp):

    ax = subplots.axes_flat[s]
    name = session.name

    legend_names.append(name)
    ses_data = means.loc[means.index.get_level_values("session") == name].mean(
        axis=0
    )  # average firing rate for each bin in a given session
    bin_duration = duration / len(ses_data)

    p = sns.lineplot(
        x=ses_data.index, y=ses_data.values, palette="hls"  # bins  # firing rates
    )

    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are on
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off

    # Plot significance indicator
    x1 = "-0.6, -0.5"  # first col
    x2 = "-0.1, 0.0005"  # last col
    p.axvspan(
        x1,
        x2,
        color="red",
        alpha=0.1,
    )

# Adjust labels
plt.ylabel("Average Firing Rate (Hz)")
plt.xlabel(
    f"Trial Duration Elapsed Relative to LED Off (as {bin_duration}s Bins)", labelpad=20
)
plt.legend(
    labels=legend_names,
    bbox_to_anchor=(1.1, 1.1),
    title="Session (RecordingDate_MouseID)",
)
plt.suptitle("Average Change In Firing Rate Relative to Grasp")
plt.axvline(
    x=round((len(ses_data.index) / 2)), ls="--", color="green"
)  # Plot vertical line at midpoint, representing the event the data is aligned to

utils.save(
    f"/home/s1735718/Figures/AllSession_Average_FiringRate_Change",
    nosize=True,
)
#%%
# Then, plot every trial, with average variance per bin (i.e., after boostrapping across unit activity)
legend_names = []
for s, session in enumerate(myexp):

    name = session.name
    legend_names.append(name)
    ses_data = within_trial_variance_change[s]

    # Determien the number of trials per session to plot as individual graphs
    ses_values = ses_data.columns.get_level_values("trial").unique()

    subplots = Subplots2D(
        ses_values, sharex=True, sharey=True
    )  # determines the number of subplots to create (as many as there are trials in the session)

    for i, value in enumerate(ses_values):
        trial_data = ses_data[value]
        ax = subplots.axes_flat[i]

        p = sns.lineplot(
            x=trial_data.columns,
            y=trial_data.loc["range"].values,
            ax=ax,
            ci=None,
            markers=None,
        )

        p.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off

        p.get_yaxis().get_label().set_visible(False)
        p.get_xaxis().get_label().set_visible(False)
        p.axvline(
            x=round((len(trial_data.columns) / 2)), ls="--", color="green"
        )  # Plot vertical line at midpoint, representing the event the data is aligned to

    # Finally, return the legend.
    legend = subplots.legend
    legend.text(0.3, 0.5, "Trial", transform=legend.transAxes)
    legend.text(
        -0.2,
        -0.5,
        "Variance in Firing Rate (95% CI)",
        transform=legend.transAxes,
        rotation=90,
    )  # y axis label
    legend.text(
        0,
        -0.3,
        f"Trial Time Elapsed Relative to LED Off (as {bin_duration}s Bins)",
        transform=legend.transAxes,
    )

    legend.set_visible(True)
    legend.get_xaxis().set_visible(False)
    legend.get_yaxis().set_visible(False)
    legend.set_facecolor("white")
    legend.set_box_aspect(1)

    plt.gcf().set_size_inches(20, 20)
    plt.show()


#%%
# Will now compare the change in variance between bins as the trial progresses.
# i.e., via another boostrapping anaysis
# Once again do this by session

cross_trial_variance = variance_boostrapping(
    within_trial_variance_change,
    myexp,
    CI=95,
    ss=20,
    bootstrap=10000,
    cache_name="variance_range_bootstrap",
    cache_overwrite=False,
)
# %%
# Run this boostrapping of variance size through the comparison
comp_cross_trial = cross_trial_bincomp(cross_trial_variance, myexp)

# Then plot changing variance with inclusion of lines denoting significance
for s, session in enumerate(myexp):
    name = session.name
    legend_names.append(name)
    ses_data = within_trial_variance_change[s]
    ses_comps = comp_cross_trial.loc[name]

    # Get only range values, convert to longform
    ses_data = ses_data.loc[ses_data.index == "range"]
    ses_data = ses_data.melt()

    # Now plot all bin values for each trial as a set of lines
    p = sns.lineplot(
        data=ses_data,
        x="bin",
        y="value",
        hue="trial",
        estimator="mean",
        lw=0.7,
    )

    p.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off

    p.axvline(
        x=round((len(ses_data.bin.unique()) / 2)), ls="--", color="green"
    )  # Plot vertical line at midpoint, representing the event the data is aligned to

    # iterate through the comparison data and where there is a significnat change, plot a shaded area
    for i, cols in enumerate(ses_comps):
        current_bin = ses_comps.index[i]
        previous_bin = ses_comps.index[i - 1]

        if cols == "increased" or cols == "decreased":
            p.axvspan(
                current_bin,
                previous_bin,
                color="red",
                alpha=0.1,
                hatch=r"/o",
            )

    plt.suptitle(f"Size of Variance per Trial for Session {name}", y=0.9)
    plt.ylabel("Variance in Firing Rate (95% CI)")
    plt.xlabel(f"Trial Time Elapsed Relative to LED Off ({bin_duration}s Bins)")
    plt.gcf().set_size_inches(10, 10)
    utils.save(
        f"/home/s1735718/Figures/{myexp[s].name}variance_change",
        nosize=True,
    )
    plt.show()

# %%
####Plot individual session FRs
legend_names = []
subplots = Subplots2D(
    myexp, sharex=True, sharey=True
)  # determines the number of subplots to create (as many as there are seaaions in the session)
plt.rcParams.update({"font.size": 30})

for s, session in enumerate(myexp):
    name = session.name
    ax = subplots.axes_flat[s]

    legend_names.append(name)
    ses_data = means.loc[
        means.index.get_level_values("session") == name
    ]  # average firing rate for each bin in a given session
    ses_data = pd.melt(ses_data.reset_index(), id_vars=["session", "trial"])
    bin_duration = duration / len(ses_data["bin"].unique())

    p = sns.lineplot(
        x=ses_data["bin"],
        y=ses_data["value"],
        palette="hls",
        ax=ax,  # bins  # firing rates
        ci="sd",
    )

    p.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are on
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off

    p.get_yaxis().get_label().set_visible(False)
    # Plot significance indicator
    if name == "211027_VR49":  # plot smaller bin for here
        x1 = "-0.2, -0.1"  # first col
        x2 = "-0.1, 0.0005"  # last col
        p.axvspan(x1, x2, color="red", alpha=0.1)
        p.axvline(x=round((len(ses_data["bin"].unique()) / 2)), ls="--", color="green")
        p.text(1, 14, name)
        p.get_xaxis().set_visible(False)
        continue

    # plot other significance values
    x1 = "-0.6, -0.5"  # first col
    x2 = "-0.1, 0.0005"  # last col
    p.axvspan(x1, x2, color="red", alpha=0.1)
    # Adjust labels

    p.get_xaxis().set_visible(False)
    p.axvline(
        x=round((len(ses_data["bin"].unique()) / 2)), ls="--", color="green"
    )  # Plot vertical line at midpoint, representing the event the data is aligned to

    # plot session number
    p.text(1, 14, name)


# Finally, return the legend.
legend = subplots.legend

legend.text(
    -0.3,
    0,
    "\n".join(wrap("Average Firing Rate (Hz)", width=20)),
    transform=legend.transAxes,
    rotation=90,
)  # y axis label
legend.text(
    0,
    -0.4,
    "\n".join(
        wrap(
            f"Trial Time Elapsed Relative to LED Off (as {bin_duration}s Bins)",
            width=25,
        )
    ),
    transform=legend.transAxes,
)

legend.set_visible(True)
legend.get_xaxis().set_visible(False)
legend.get_yaxis().set_visible(False)
legend.set_facecolor("white")
legend.set_box_aspect(1)

plt.suptitle("Average Firing Rate Per Session", y=0.95)
plt.gcf().set_size_inches(25, 25)
plt.rcParams.update({"font.size": 30})
utils.save(
    f"/home/s1735718/Figures/AllSession_Average_FiringRate_Change_individualPlots",
    nosize=True,
)

# %%
