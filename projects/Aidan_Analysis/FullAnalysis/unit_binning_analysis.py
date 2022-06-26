#################################################################################################################
# As described in my notes, this analysis shall perform the following:
# 1. Take data from entire trials across units (i.e., aligned to LED off)
# 2. Normalise each trials data as a percentage of the total time elapsed
# 3. Bin this data into 10% bins
# 4. Compare each bin to the previous, with the exception of the first bin, where no comparison shall take place
# 5. Create a histogram of the number of significant units across general trial space (i.e., as a normalised %)
#################################################################################################################

##TODO: Split the two analyses in this file into seperate scripts, will make things easier to manage
##TODO: Rerun this code on only M2 and see if this changes findings.

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
from CI_Analysis import significance_extraction
from CI_Analysis import percentile_plot
from functions import event_times, per_trial_spike_rate

sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")
sys.path.insert(0, "/home/s1735718/PixelsAnalysis/pixels/pixels")
from pixtools.utils import Subplots2D  # use the local copy of base.py
from pixtools import utils
from pixtools import spike_rate
from pixels import ioutils


# Now select the units that shall be analysed


units = myexp.select_units(group="good", name="all_layers", max_depth=3500)

# select all pyramidal neurons
# pyramidal_units = myexp.select_units(
#     group="good", min_depth=200, max_depth=1200, name="pyramidals", min_spike_width=0.4
# )

# interneuron_units = myexp.select_units(
#     group="good",
#     min_depth=200,
#     max_depth=1200,
#     name="interneurons",
#     max_spike_width=0.35,
# )

# Then align trials to LED off (i.e, end of trial) with the time window determined by the max trial length
duration = 10
trial_aligned = myexp.align_trials(
    ActionLabels.correct, Events.led_off, "spike_rate", duration=duration, units=units
)

# pyramidal_aligned = myexp.align_trials(
#     ActionLabels.correct,
#     Events.led_off,
#     "spike_rate",
#     duration=10,
#     units=pyramidal_units,
# )

# interneuron_aligned = myexp.align_trials(
#     ActionLabels.correct,
#     Events.led_off,
#     "spike_rate",
#     duration=10,
#     units=interneuron_units,
# )

#%%
##Now shall bin the data within each trial into 100 ms bins
# Will create a function to do so:


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


#%%
##Now shall create a function to take this bin information and conduct a boostrapping analysis
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


#%%
##Now shall create a function to take these averaged bin values and perform a bootstrapping analysis
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


#%%
##Finally, will create a function to take these percentiles and compare them.
# Specifically, per trial, each bin will be compared to the one previous
# If 2.5% and previous 97.5% overlap then there is a significant change
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


#%%
##Run these functions to perform the binning and boostrapping of data##
# Uncomment to run
# bin_data = per_trial_binning(trial_aligned, myexp, timespan=10, bin_size=0.1)
# bin_avgs = bin_data_average(bin_data)
# bootstrap = bin_data_bootstrap(
#     bin_avgs, CI=95, ss=20, bootstrap=10000, name="fullbootstrap", cache_overwrite=False
# )
# bin_comp = bs_pertrial_bincomp(bootstrap, myexp)

# %%

#%%
# #### Now may plot changing spikerates for each trial including the point at which LED turned on (i.e., the point where trial began)
# # First determine the times LED turned on for each trial
ledon = event_times("led_on", myexp)
ledon = pd.DataFrame(ledon)

# Then when LED turned off
ledoff = event_times("led_off", myexp)
ledoff = pd.DataFrame(ledoff)

# Then determine the time relative to LED off (alignment point) where led turned on for each trial
start = ledon - ledoff
start = {ses.name: start.loc[s] for s, ses in enumerate(myexp)}
start = pd.concat(start, axis=1)
start = start / 1000  # Convert from ms to s
start.index.name = "trial"  # Rename the index to trial number (ascending order)

# Now check what the max length of the trial is
vals = start.values.copy()
vals = vals.astype(int)  # Convert nan values to integer (a very large negative value)
vals[vals < -90000000] = 0  # Convert all large negative values to zero
print(np.abs(vals).max())  # Check the max value with nan removed


## Plot a single graph for every session
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


# Uncomment below to plot this
# bs_graph(trial_aligned, bin_data, bin_comp, myexp)
