# First import required packages
from argparse import Action
from curses import raw
import math

from matplotlib.pyplot import axvline, xlim
from base import *
import numpy as np
import pandas as pd
import matplotlib as plt
from pixtools.utils import Subplots2D
from rasterbase import *
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from pixtools.utils import Subplots2D
from pixels import ioutils

from functions import (
    event_times,
    per_trial_raster,
    per_trial_binning,
    per_unit_spike_rate,
    bin_data_average,
)

#%%
# First, select only units from correct trials, of good quality, and within m2
units = myexp.select_units(group="good", name="m2", min_depth=200, max_depth=1200)

l23_units = myexp.select_units(
    group="good", name="layerII/III", min_depth=200, max_depth=375
)
l5_units = myexp.select_units(group="good", name="layerV", min_depth=375, max_depth=815)
l6_units = myexp.select_units(
    group="good", name="layerVI", min_depth=815, max_depth=1200
)

# Now align these firing rates to the point at which the trial began
duration = 8

# Specifically, to calculate FANO FACTOR we need to get the exact spike times.
spike_times = myexp.align_trials(
    ActionLabels.clean,
    Events.reach_onset,
    "spike_times",
    duration=duration,
    units=units,
)
spike_times = spike_times.reorder_levels(
    ["session", "trial", "unit"], axis=1
)  # reorder the levels

l23_spike_times = myexp.align_trials(
    ActionLabels.clean,
    Events.reach_onset,
    "spike_times",
    duration=duration,
    units=l23_units,
)
l23_spike_times = l23_spike_times.reorder_levels(
    ["session", "trial", "unit"], axis=1
)  # reorder the levels

l5_spike_times = myexp.align_trials(
    ActionLabels.clean,
    Events.reach_onset,
    "spike_times",
    duration=duration,
    units=l5_units,
)
l5_spike_times = l5_spike_times.reorder_levels(
    ["session", "trial", "unit"], axis=1
)  # reorder the levels

l6_spike_times = myexp.align_trials(
    ActionLabels.clean,
    Events.reach_onset,
    "spike_times",
    duration=duration,
    units=l6_units,
)
l6_spike_times = l6_spike_times.reorder_levels(
    ["session", "trial", "unit"], axis=1
)  # reorder the levels

firing_rates = myexp.align_trials(
    ActionLabels.clean, Events.reach_onset, "spike_rate", duration=duration, units=units
)
firing_rate_means = per_trial_binning(
    firing_rates, myexp, timespan=duration, bin_size=0.1
)
firing_rate_means = bin_data_average(firing_rate_means)
firing_rate_means = firing_rate_means.mean()
# l23_firing_rates = myexp.align_trials(
#     ActionLabels.clean,
#     Events.reach_onset,
#     "spike_rate",
#     duration=duration,
#     units=l23_units,
# )
# l23_firing_rates = l23_firing_rates.reorder_levels(
#     ["session", "unit", "trial"], axis=1
# )  # reorder the levels

# l5_firing_rates = myexp.align_trials(
#     ActionLabels.clean,
#     Events.reach_onset,
#     "spike_rate",
#     duration=duration,
#     units=l5_units,
# )
# l5_firing_rates = l5_firing_rates.reorder_levels(
#     ["session", "unit", "trial"], axis=1
# )  # reorder the levels

# l6_firing_rates = myexp.align_trials(
#     ActionLabels.clean,
#     Events.reach_onset,
#     "spike_rate",
#     duration=duration,
#     units=l6_units,
# )
# l6_firing_rates = l6_firing_rates.reorder_levels(
#     ["session", "unit", "trial"], axis=1
# )  # reorder the levels

# Take average firing rates
# l23_firing_rates = per_trial_binning(
#     l23_firing_rates, myexp, timespan=duration, bin_size=0.1
# )
# l23_means = bin_data_average(l23_firing_rates)
# l23_means = l23_means.mean()

# l5_firing_rates = per_trial_binning(
#     l5_firing_rates, myexp, timespan=duration, bin_size=0.1
# )
# l5_means = bin_data_average(l5_firing_rates)
# l5_means = l5_means.mean()

# l6_firing_rates = per_trial_binning(
#     l6_firing_rates, myexp, timespan=duration, bin_size=0.1
# )
# l6_means = bin_data_average(l6_firing_rates)
# l6_means = l6_means.mean()

# From this, must now calculate the point at which the LED turned off in each trial.
# Will allow plotting of off time.
off_times = event_times("led_off", myexp)  # time after LED on when trial finished

# %%
# Now shall define a function to calculate per Trial FF


def cross_trial_FF(
    spike_times,
    exp,
    duration,
    bin_size,
    cache_name,
    ff_iterations=50,
    mean_matched=False,
    raw_values=False,
):
    """
    This function will take raw spike times calculated through the exp.align_trials method and calculate the fano factor for each trial, cross units
    or for individual units, if mean matched is False.

    ------------------------------------------------------------------------
    |NB: Output of the function is saved to an h5 file as "FF_Calculation" |
    ------------------------------------------------------------------------

    spike_times: The raw spike time data produced by the align_trials method

    exp: the experimental cohort defined in base.py (through the Experiment class)

    duration: the window of the aligned data (in s)

    bin_size: the size of the bins to split the trial duration by (in s)

    name: the name to cache the results under

    mean_matched: Whether to apply mean matching adjustment (as defined by Churchland 2010) to correct for large changes in unit mean firing rates across trials
                  If this is False, the returned dataframe will be of unadjusted single unit FFs

    raw_values: Whether to return the raw values rather than mean FFs

    ff_iterations: The number of times to repeat the random mean-matched sampling when calculating population fano factor (default 50)

    """
    # First create bins that the data shall be categorised into
    duration = duration * 1000  # convert s to ms
    bin_size = bin_size * 1000
    bins = np.arange((-duration / 2), (duration / 2), bin_size)
    all_FF = pd.DataFrame()

    # First split data by session
    for s, session in enumerate(myexp):
        ses_data = spike_times[s]
        ses_data = ses_data.reorder_levels(["unit", "trial"], axis=1)
        name = session.name
        session_FF = {}
        session_se = {}
        session_ci = {}

        repeat_FF = pd.DataFrame()
        repeat_se = pd.DataFrame()
        repeat_ci = pd.DataFrame()

        # Bin the session data
        ses_vals = ses_data.values
        bin_ids = np.digitize(ses_vals[~np.isnan(ses_vals)], bins, right=True)

        # Then by unit
        unit_bin_means = []
        unit_bin_var = []

        print(f"beginning mean/variance calculation for session {name}")
        keys = []
        for t in tqdm(ses_data.columns.get_level_values("unit").unique()):
            unit_data = ses_data[t]  # take each unit values for all trials
            unit_data = (
                unit_data.melt().dropna()
            )  # Pivot this data and drop missing values

            # Once these values have been isolated, can begin FF calculation.
            unit_data["bin"] = np.digitize(unit_data["value"], bins, right=True)

            bin_count_means = []
            bin_count_var = []

            for b in np.unique(bin_ids):
                bin_data = unit_data.loc[unit_data["bin"] == b]

                # Calculate the count mean of this bin, i.e., the average number of events per trial
                bin_count_mean = (
                    bin_data.groupby("trial").count().mean()["value"]
                )  # the average number of spikes per unit in this bin

                # If there are any bins where no spikes occurred (i.e, empty dataframes) then return zero
                if np.isnan(bin_count_mean):
                    bin_count_mean = 0

                bin_count_means.append(bin_count_mean)

                # Now calculate variance of the event count
                bin_var = np.var(bin_data.groupby("trial").count()["value"])

                # Again return zero if no events
                if np.isnan(bin_var):
                    bin_var = 0

                bin_count_var.append(bin_var)

            # These values will make up a single point on the scatterplots used to calculate cross-unit FF
            # Append to list containing values for each unit, across trials

            unit_bin_means.append(bin_count_means)
            unit_bin_var.append(bin_count_var)
            keys.append(t)

        # Now convert this to a dataframe
        unit_bin_means = pd.DataFrame.from_records(unit_bin_means, index=keys)
        unit_bin_means.index.name = "unit"
        unit_bin_means.columns.name = "bin"

        unit_bin_var = pd.DataFrame.from_records(unit_bin_var, index=keys)
        unit_bin_var.index.name = "unit"
        unit_bin_var.columns.name = "bin"

        # can now use this information to calculate FF, and include any potential adjustments needed.
        # To do so, shall create a linear regression to calculate FF, for each bin

        if mean_matched is True:
            bin_heights = {}
            """
            First calculate the greatest common distribution for this session's data
            1. Calculate the distribution of mean counts for each timepoint (binned trial time)
            2. Take the smallest height of each bin across timepoints, this will form the height of the GCD's corresponding bin
            3. Save this GCD to allow further mean matching
            """

            # Take each bin, and unit, then calculate the distribution of mean counts
            for b in unit_bin_means.columns:

                # Isolate bin values, across units
                bin_means = unit_bin_means[b]

                # Calculate distribution for this time, across units
                p = np.histogram(bin_means, bins=9)

                # Save the heights of these bins for each bin
                bin_heights.update(
                    {b: p[0]}
                )  # append to a dictionary, where the key is the bin

            # Convert this dictionary to a dataframe, containing the heights of each timepoints distribution
            bin_heights = pd.DataFrame.from_dict(
                bin_heights, orient="index"
            )  # the index of this dataframe will be the binned timepoints
            gcd_heights = (
                bin_heights.min()
            )  # get the minimum value for each "bin" of the histogram across timepoints
            gcd_heights.index += 1

            # Now may begin mean matching to the gcd
            # Take each bin (i.e., timepoint) and calculate the distribution of points
            # TODO: Update this analysis to utilise a rolling bin approach. I.e, take data at 100ms intervals, then shift the bin 10ms right

            print(f"Initiating Mean Matched Fano Factor Calculation for Session {name}")
            for iteration in tqdm(range(ff_iterations)):
                np.random.seed(iteration)

                for b in unit_bin_means.columns:
                    timepoint_means = pd.DataFrame(unit_bin_means[b])
                    timepoint_edges = pd.DataFrame(
                        np.histogram(timepoint_means, bins=9)[1]
                    )  # the left edges of the histogram used to create a distribution
                    timepoint_var = unit_bin_var[b]

                    # split data into bins
                    binned_data = pd.cut(
                        np.ndarray.flatten(timepoint_means.values),
                        np.ndarray.flatten(timepoint_edges.values),
                        labels=timepoint_edges.index[1:],
                        include_lowest=True,
                    )  # seperate raw data into the distribution bins

                    timepoint_means["bin"] = binned_data

                    # Isolate a single bin from the timepoint histogram
                    timepoint_FF_mean = {}
                    timepoint_FF_var = {}

                    for i in range(len(timepoint_edges.index)):
                        if i == 0:
                            continue
                        # Bins begin at one, skip first iteration
                        repeat = True
                        bin_data = timepoint_means.loc[timepoint_means["bin"] == i]

                        while repeat == True:

                            # Check if bin height matches gcd height - if it does, append timepoint data to list

                            if bin_data.count()["bin"] == gcd_heights[i]:
                                timepoint_FF_mean.update(
                                    {i: bin_data[bin_data.columns[0]]}
                                )  # append the bin, and unit/mean count
                                repeat = False

                            # If not, remove randomly points from the dataset one at a time, until they match
                            else:
                                dropped_data = np.random.choice(
                                    bin_data.index, 1, replace=False
                                )
                                bin_data = bin_data.drop(
                                    dropped_data
                                )  # remove one datapoint, then continue with check

                    ##Now extract the associated variance for the timepoint
                    timepoint_FF_mean = pd.DataFrame.from_dict(timepoint_FF_mean)
                    timepoint_FF_mean = timepoint_FF_mean.melt(
                        ignore_index=False
                    ).dropna()
                    timepoint_FF_var = timepoint_var[
                        timepoint_var.index.isin(timepoint_FF_mean.index)
                    ]

                    # Using the count mean and variance of the count, construct a linear regression for the population
                    # Ensure the origin of the linear model is set as zero

                    # Set up data in correct shape
                    x = np.array(timepoint_FF_mean["value"]).reshape(
                        (-1, 1)
                    )  # mean count, the x axis
                    y = np.array(timepoint_FF_var)

                    # Now fit model
                    timepoint_model = sm.OLS(y, x).fit()

                    # The slope (i.e., coefficient of variation) of this model
                    ff_score = timepoint_model.params[0]

                    # Extract the confidence interval of this model fit
                    ff_se = timepoint_model.bse[0]
                    ff_ci = timepoint_model.conf_int()[0]

                    session_FF.update({b: ff_score})
                    session_se.update({b: ff_se})
                    session_ci.update({b: tuple(ff_ci)})

                # Convert the session's FF to dataframe, then add to a master
                session_FF = pd.DataFrame(session_FF, index=[0])
                session_se = pd.DataFrame(session_se, index=[0])
                session_ci = pd.DataFrame(session_ci.items())

                repeat_FF = pd.concat([repeat_FF, session_FF], axis=0)
                repeat_se = pd.concat([repeat_se, session_se], axis=0)
                repeat_ci = pd.concat([repeat_ci, session_ci], axis=0)

                session_FF = {}
                session_se = {}
                session_ci = {}

            # Take the average across repeats for each session, add this to a final dataframe for session, timepoint, mean_FF, SE

            for i, cols in enumerate(repeat_FF.iteritems()):
                timepoint_vals = repeat_FF.describe()[i]
                timepoint_se = repeat_se.describe()[i]

                ci_bins = repeat_ci[0].tolist()
                timepoint_ci = pd.DataFrame(repeat_ci[1].tolist())
                timepoint_ci = timepoint_ci.rename(
                    columns={0: "lower_ci", 1: "upper_ci"}
                ).reset_index(drop=True)
                timepoint_ci.insert(0, "bin", ci_bins)

                mean = timepoint_vals["mean"]
                se = timepoint_se["mean"]
                lower_ci = timepoint_ci.loc[timepoint_ci["bin"] == i].describe()[
                    "lower_ci"
                ]["mean"]
                upper_ci = timepoint_ci.loc[timepoint_ci["bin"] == i].describe()[
                    "upper_ci"
                ]["mean"]

                if raw_values == False:
                    vals = pd.DataFrame(
                        {
                            "session": name,
                            "bin": i,
                            "mean FF": mean,
                            "SE": se,
                            "lower_ci": lower_ci,
                            "upper_ci": upper_ci,
                        },
                        index=[0],
                    )
                    all_FF = pd.concat([all_FF, vals], ignore_index=True)
                elif raw_values == True:
                    repeat_FF = repeat_FF.melt()
                    repeat_se = repeat_se.melt()

                    repeat_FF["session"] = name
                    repeat_FF = repeat_FF.set_index("session")
                    repeat_FF = repeat_FF.rename(
                        columns={"value": "FF", "variable": "bin"}
                    )

                    repeat_se["session"] = name
                    repeat_se = repeat_se.set_index("session")
                    repeat_FF["se"] = repeat_se["value"]
                    repeat_FF["lower_ci"] = timepoint_ci["lower_ci"]
                    repeat_FF["upper_ci"] = timepoint_ci["upper_ci"]

                    all_FF = pd.concat([all_FF, repeat_FF])
                    break

        elif mean_matched is False:
            # Remember, this will return the individual FF for each unit, not the population level FF
            # calculate raw FF for each unit
            session_FF = pd.DataFrame(columns=["bin", "FF", "session", "unit"])
            for unit in unit_bin_means.index:
                unit_FF = pd.DataFrame(
                    unit_bin_var.loc[unit] / unit_bin_means.loc[unit]
                ).rename(
                    columns={unit: "FF"}
                )  # return all bins for this unit
                unit_FF = unit_FF.reset_index()
                unit_FF["session"] = name
                unit_FF["unit"] = unit

                session_FF = pd.concat([session_FF, unit_FF])

            # Reorder columns, then return data
            session_FF = session_FF[["session", "unit", "bin", "FF"]]
            session_FF.reset_index(drop=True)
            all_FF = pd.concat([all_FF, session_FF], axis=0)

    ioutils.write_hdf5(
        f"/data/aidan/{cache_name}_FF_Calculation_mean_matched_{mean_matched}_raw{raw_values}.h5",
        all_FF,
    )
    return all_FF


# test = cross_trial_FF(
#     spike_times=spike_times,
#     exp=myexp,
#     duration=4,
#     bin_size=0.1,
#     cache_name="per_unit_FF",
#     mean_matched=True,
#     ff_iterations=2,
# )


# %%
# Plot individual unit FFs for each session overlaid on firing rates
# Split this by unit type

# l23_per_unit_FF = cross_trial_FF(
#     spike_times=l23_spike_times,
#     exp=myexp,
#     duration=4,
#     bin_size=0.1,
#     cache_name="per_unit_FF",
#     mean_matched=False,
#     ff_iterations=50,
# )

# l5_per_unit_FF = cross_trial_FF(
#     spike_times=l5_spike_times,
#     exp=myexp,
#     duration=4,
#     bin_size=0.1,
#     cache_name="per_unit_FF",
#     mean_matched=False,
#     ff_iterations=50,
# )

# l6_per_unit_FF = cross_trial_FF(
#     spike_times=l6_spike_times,
#     exp=myexp,
#     duration=4,
#     bin_size=0.1,
#     cache_name="per_unit_FF",
#     mean_matched=False,
#     ff_iterations=50,
# )
# # Concatenate data
# l23_per_unit_FF["layer"] = "layer 2/3"
# l5_per_unit_FF["layer"] = "layer 5"
# l6_per_unit_FF["layer"] = "layer 6"

# per_unit_FF = pd.concat(
#     [l23_per_unit_FF, l5_per_unit_FF, l6_per_unit_FF], ignore_index=True
# # )

# # First plot layer 2/3
# for s, session in enumerate(myexp):
#     name = session.name
#     units = l23_per_unit_FF["unit"].unique()
#     subplots = Subplots2D(units, sharex=False)
#     layer = "23"
#     palette = sns.color_palette()
#     plt.rcParams["figure.figsize"] = (10, 10)
#     ses_fr = l23_firing_rates[s]

#     for i, un in enumerate(units):
#         unit_data = l23_per_unit_FF.loc[l23_per_unit_FF["unit"] == un]
#         unit_data["bin/10"] = unit_data["bin"] / 10 - (
#             duration / 2
#         )  # convert 100ms bins to seconds relative to action
#         ax = subplots.axes_flat[i]
#         ax1 = ax.twinx()  # secondary axis for fr

#         bins = np.unique(
#             l23_per_unit_FF.loc[l23_per_unit_FF["session"] == name]["bin"].values
#         )

#         val_data = ses_fr[un].stack().reset_index()
#         val_data["y"] = val_data[0]
#         # num_samples = len(ses_fr[values[0]].columns)

#         p = sns.lineplot(
#             data=unit_data, x="bin/10", y="FF", ax=ax, linewidth=0.5, color="red"
#         )
#         p.axvline(
#             len(np.unique(bins)) / 2,
#             color="green",
#             ls="--",
#         )
#         # p.autoscale(enable=True, tight=False)
#         p.set_xticks([])
#         p.set_yticks([])
#         p.get_yaxis().get_label().set_visible(False)
#         p.get_xaxis().get_label().set_visible(False)
#         ax.set_xlim(bins[0], bins[-1])

#         q = sns.lineplot(
#             data=val_data,
#             x="time",
#             y="y",
#             ci="sd",
#             ax=ax1,
#             linewidth=0.5,
#         )
#         # q.autoscale(enable=True, tight=False)
#         q.set_yticks([])
#         q.set_xticks([])
#         ax1.set_xlim(-duration / 2, duration / 2)
#         # peak = ses_fr[value].values.mean(axis=1).max()
#         q.get_yaxis().get_label().set_visible(False)
#         q.get_xaxis().get_label().set_visible(False)
#         q.axvline(c=palette[1], ls="--", linewidth=0.5)
#         q.set_box_aspect(1)

#     to_label = subplots.axes_flat[0]
#     to_label.get_yaxis().get_label().set_visible(True)
#     to_label.get_xaxis().get_label().set_visible(True)
#     to_label.set_xticks([])

#     legend = subplots.legend

#     legend.text(
#         0,
#         0.3,
#         "Unit ID",
#         transform=legend.transAxes,
#         color=palette[0],
#     )
#     legend.set_visible(True)
#     ticks = list(unit_data["bin"].unique())
#     legend.get_xaxis().set_ticks(
#         ticks=[ticks[0], ticks[round(len(ticks) / 2)], ticks[-1]]
#     )
#     legend.set_xlabel("100ms Bin")
#     legend.get_yaxis().set_visible(False)
#     legend.set_box_aspect(1)

#     # Overlay firing rates
#     values = ses_fr.columns.get_level_values("unit").unique()
#     values.values.sort()
#     # num_samples = len(ses_fr[values[0]].columns)

#     plt.suptitle(f"Session {name} Per Unit Fano Factor + Firing Rate")

#     utils.save(
#         f"/home/s1735718/Figures/{name}_per_unit_FanoFactor_l{layer}", nosize=True
#     )

# for s, session in enumerate(myexp):
#     name = session.name
#     units = l5_per_unit_FF["unit"].unique()
#     subplots = Subplots2D(units, sharex=False)
#     layer = "5"
#     palette = sns.color_palette()
#     plt.rcParams["figure.figsize"] = (10, 10)
#     ses_fr = l5_firing_rates[s]

#     for i, un in enumerate(units):
#         unit_data = l5_per_unit_FF.loc[l5_per_unit_FF["unit"] == un]
#         unit_data["bin/10"] = unit_data["bin"] / 10 - (
#             duration / 2
#         )  # convert 100ms bins to seconds relative to action
#         ax = subplots.axes_flat[i]
#         ax1 = ax.twinx()  # secondary axis for fr

#         bins = np.unique(
#             l5_per_unit_FF.loc[l5_per_unit_FF["session"] == name]["bin"].values
#         )

#         val_data = ses_fr[un].stack().reset_index()
#         val_data["y"] = val_data[0]
#         # num_samples = len(ses_fr[values[0]].columns)

#         p = sns.lineplot(
#             data=unit_data, x="bin/10", y="FF", ax=ax, linewidth=0.5, color="red"
#         )
#         p.axvline(
#             len(np.unique(bins)) / 2,
#             color="green",
#             ls="--",
#         )
#         # p.autoscale(enable=True, tight=False)
#         p.set_xticks([])
#         p.set_yticks([])
#         p.get_yaxis().get_label().set_visible(False)
#         p.get_xaxis().get_label().set_visible(False)
#         ax.set_xlim(bins[0], bins[-1])

#         q = sns.lineplot(
#             data=val_data,
#             x="time",
#             y="y",
#             ci="sd",
#             ax=ax1,
#             linewidth=0.5,
#         )
#         # q.autoscale(enable=True, tight=False)
#         q.set_yticks([])
#         q.set_xticks([])
#         ax1.set_xlim(-duration / 2, duration / 2)
#         # peak = ses_fr[value].values.mean(axis=1).max()
#         q.get_yaxis().get_label().set_visible(False)
#         q.get_xaxis().get_label().set_visible(False)
#         q.axvline(c=palette[1], ls="--", linewidth=0.5)
#         q.set_box_aspect(1)

#     to_label = subplots.axes_flat[0]
#     to_label.get_yaxis().get_label().set_visible(True)
#     to_label.get_xaxis().get_label().set_visible(True)
#     to_label.set_xticks([])

#     legend = subplots.legend

#     legend.text(
#         0,
#         0.3,
#         "Unit ID",
#         transform=legend.transAxes,
#         color=palette[0],
#     )
#     legend.set_visible(True)
#     ticks = list(unit_data["bin"].unique())
#     legend.get_xaxis().set_ticks(
#         ticks=[ticks[0], ticks[round(len(ticks) / 2)], ticks[-1]]
#     )
#     legend.set_xlabel("100ms Bin")
#     legend.get_yaxis().set_visible(False)
#     legend.set_box_aspect(1)

#     # Overlay firing rates
#     values = ses_fr.columns.get_level_values("unit").unique()
#     values.values.sort()
#     # num_samples = len(ses_fr[values[0]].columns)

#     plt.suptitle(f"Session {name} Per Unit Fano Factor + Firing Rate")

#     utils.save(
#         f"/home/s1735718/Figures/{name}_per_unit_FanoFactor_l{layer}", nosize=True
#     )

# for s, session in enumerate(myexp):
#     name = session.name
#     units = l6_per_unit_FF["unit"].unique()
#     subplots = Subplots2D(units, sharex=False)
#     layer = "6"
#     palette = sns.color_palette()
#     plt.rcParams["figure.figsize"] = (10, 10)
#     ses_fr = l6_firing_rates[s]

#     for i, un in enumerate(units):
#         unit_data = l6_per_unit_FF.loc[l6_per_unit_FF["unit"] == un]
#         unit_data["bin/10"] = unit_data["bin"] / 10 - (
#             duration / 2
#         )  # convert 100ms bins to seconds relative to action
#         ax = subplots.axes_flat[i]
#         ax1 = ax.twinx()  # secondary axis for fr

#         bins = np.unique(
#             l6_per_unit_FF.loc[l6_per_unit_FF["session"] == name]["bin"].values
#         )

#         val_data = ses_fr[un].stack().reset_index()
#         val_data["y"] = val_data[0]
#         # num_samples = len(ses_fr[values[0]].columns)

#         p = sns.lineplot(
#             data=unit_data, x="bin/10", y="FF", ax=ax, linewidth=0.5, color="red"
#         )
#         p.axvline(
#             len(np.unique(bins)) / 2,
#             color="green",
#             ls="--",
#         )
#         # p.autoscale(enable=True, tight=False)
#         p.set_xticks([])
#         p.set_yticks([])
#         p.get_yaxis().get_label().set_visible(False)
#         p.get_xaxis().get_label().set_visible(False)
#         ax.set_xlim(bins[0], bins[-1])

#         q = sns.lineplot(
#             data=val_data,
#             x="time",
#             y="y",
#             ci="sd",
#             ax=ax1,
#             linewidth=0.5,
#         )
#         # q.autoscale(enable=True, tight=False)
#         q.set_yticks([])
#         q.set_xticks([])
#         ax1.set_xlim(-duration / 2, duration / 2)
#         # peak = ses_fr[value].values.mean(axis=1).max()
#         q.get_yaxis().get_label().set_visible(False)
#         q.get_xaxis().get_label().set_visible(False)
#         q.axvline(c=palette[1], ls="--", linewidth=0.5)
#         q.set_box_aspect(1)

#     to_label = subplots.axes_flat[0]
#     to_label.get_yaxis().get_label().set_visible(True)
#     to_label.get_xaxis().get_label().set_visible(True)
#     to_label.set_xticks([])

#     legend = subplots.legend

#     legend.text(
#         0,
#         0.3,
#         "Unit ID",
#         transform=legend.transAxes,
#         color=palette[0],
#     )
#     legend.set_visible(True)
#     ticks = list(unit_data["bin"].unique())
#     legend.get_xaxis().set_ticks(
#         ticks=[ticks[0], ticks[round(len(ticks) / 2)], ticks[-1]]
#     )
#     legend.set_xlabel("100ms Bin")
#     legend.get_yaxis().set_visible(False)
#     legend.set_box_aspect(1)

#     # Overlay firing rates
#     values = ses_fr.columns.get_level_values("unit").unique()
#     values.values.sort()
#     # num_samples = len(ses_fr[values[0]].columns)

#     plt.suptitle(f"Session {name} Per Unit Fano Factor + Firing Rate")

#     utils.save(
#         f"/home/s1735718/Figures/{name}_per_unit_FanoFactor_l{layer}", nosize=True
#     )

# # %%
# # Plot population FF changes

# population_FF_l23 = cross_trial_FF(
#     spike_times=l23_spike_times,
#     exp=myexp,
#     duration=4,
#     bin_size=0.1,
#     cache_name="l23",
#     mean_matched=True,
#     ff_iterations=50,
# )
# population_FF_l5 = cross_trial_FF(
#     spike_times=l5_spike_times,
#     exp=myexp,
#     duration=4,
#     bin_size=0.1,
#     cache_name="l5",
#     mean_matched=True,
#     ff_iterations=50,
# )
# population_FF_l6 = cross_trial_FF(
#     spike_times=l6_spike_times,
#     exp=myexp,
#     duration=4,
#     bin_size=0.1,
#     cache_name="l6",
#     mean_matched=True,
#     ff_iterations=50,
# )

#%%
# # Plot lines for each layer
# fig, ax = plt.subplots(2, figsize=(10, 10))
# bin_size = 0.1
# ax1 = ax[0]
# # First layer 2/3
# mean = population_FF_l23["mean FF"]
# se = population_FF_l23["SE"]
# x = (population_FF_l23["bin"] * bin_size) - 2
# lower = mean - se
# upper = mean + se

# ax1.plot(x, mean, label="Layer 2/3", color="blue")
# ax1.plot(x, lower, color="tab:blue", alpha=0.1)
# ax1.plot(x, upper, color="tab:blue", alpha=0.1)
# ax1.fill_between(x, lower, upper, alpha=0.2)

# # Then layer 5
# mean = population_FF_l5["mean FF"]
# se = population_FF_l5["SE"]
# x = (population_FF_l5["bin"] * bin_size) - 2
# lower = mean - se
# upper = mean + se

# ax1.plot(x, mean, label="Layer 5", color="orange")
# ax1.plot(x, lower, color="tab:orange", alpha=0.1)
# ax1.plot(x, upper, color="tab:orange", alpha=0.1)
# ax1.fill_between(x, lower, upper, alpha=0.2)

# # Finally, layer 6
# mean = population_FF_l6["mean FF"]
# se = population_FF_l6["SE"]
# x = (population_FF_l6["bin"] * bin_size) - 2
# lower = mean - se
# upper = mean + se

# ax1.plot(x, mean, label="Layer 6", color="green")
# ax1.plot(x, lower, color="tab:green", alpha=0.1)
# ax1.plot(x, upper, color="tab:green", alpha=0.1)
# ax1.fill_between(x, lower, upper, alpha=0.2)

# # Now set axes
# ax1.set_ylabel("Mean Matched Fano Factor")
# ax1.set_xticklabels([])
# ax1.axvline(x=0, color="black", ls="--")
# ax1.set_ylim(0)
# ax1.legend()


# plt.suptitle(
#     f"Population Level Fano Factor - Aligned to Reach Onset - Session {myexp[0].name}",
#     y=0.9,
# )

# # Now plot firing rates
# ax2 = ax[1]

# ax2.plot(x, l23_means, color="blue")
# ax2.plot(x, l5_means, color="orange")
# ax2.plot(x, l6_means, color="green")
# ax2.axvline(x=0, color="black", ls="--")
# ax2.set_title("Population Level Firing Rate")
# ax2.set_ylabel("Mean Firing Rate (Hz)")
# ax2.set_xlabel("Time Relative to Reach Onset (s)")
# ax2.set_ylim(0)
# utils.save(f"/home/s1735718/Figures/{myexp[0].name}_population_FanoFactor", nosize=True)
# plt.show()
# %%
# Plot undivided fano factor for the population
population_FF = cross_trial_FF(
    spike_times=spike_times,
    exp=myexp,
    duration=8,
    bin_size=0.1,
    cache_name="population",
    mean_matched=True,
    ff_iterations=50,
)
# Check significance for each bin


def fano_fac_sig(population_FF):
    """
    Function checks for significance in fano factor output, assuming the data is NOT raw, and mean-matched.
    Returns a list of significance values (sig or nonsig) for each bin, compared to the previous timepoint.

    population_FF: the data obtained from the cross_trial_FF() function, where mean_matched is True, and raw_data is False.
    """
    significance = []
    for i, items in enumerate(population_FF.iterrows()):
        if i == 0:
            continue

        bin_upper = items[1]["upper_ci"]
        bin_lower = items[1]["lower_ci"]

        previous_upper = population_FF.loc[population_FF["bin"] == i - 1]["upper_ci"]
        previous_lower = population_FF.loc[population_FF["bin"] == i - 1]["lower_ci"]

        # Compare bins for overlap
        if (bin_upper <= previous_lower.item()) or (bin_lower >= previous_upper.item()):
            significance.append("significant")

        else:
            significance.append("nonsignificant")

    return significance

significance = fano_fac_sig(population_FF)
#%%
# Plot firing rates and fano factor
name = myexp[0].name
bin_size = 0.1
bins = per_trial_binning(firing_rates, myexp, timespan=duration, bin_size=0.1)
bins = bin_data_average(bins).describe()
std = []
for i in bins.iteritems():
    std.append(i[1]["std"])

fig, ax = plt.subplots(2, figsize=(10, 10))
ax1 = ax[0]
ax2 = ax[1]

mean = population_FF["mean FF"]
se = population_FF["SE"]
upper = mean + se
lower = mean - se
x = (population_FF["bin"] * bin_size) - 4

ax1.plot(x, mean, color="purple")
ax1.plot(x, lower, color="tab:purple", alpha=0.1)
ax1.plot(x, upper, color="tab:purple", alpha=0.1)
ax1.fill_between(x, lower, upper, alpha=0.2)

ax1.set_ylabel("Mean Matched Fano Factor")
ax1.set_xlabel("Duration Relative to Reach Onset (s)")
ax1.axvline(x=0, color="black", ls="--")
ax1.set_ylim(0)
ax1.set_title(f"Population Level Fano Factor - session {name}")

# Plot firing rate
lower = firing_rate_means - std
upper = firing_rate_means + std

ax2.plot(x, firing_rate_means)
ax2.plot(x, lower, color="tab:purple", alpha=0.1)
ax2.plot(x, upper, color="tab:purple", alpha=0.1)
ax2.fill_between(x, lower, upper, alpha=0.2)

ax2.set_title("Population Level Firing Rate")
ax2.set_ylabel("Mean Firing Rate (Hz)")
ax2.set_xlabel("Time Relative to Reach Onset (s)")

ax2.axvline(x=0, color="black", ls="--")

utils.save(
    f"/home/s1735718/Figures/{myexp[0].name}_population_FanoFactor_unsplit", nosize=True
)
plt.show()
# %%
