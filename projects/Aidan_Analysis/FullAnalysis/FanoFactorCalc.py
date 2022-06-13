# First import required packages
from argparse import Action

from matplotlib.pyplot import xlim
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
)

#%%
# First, select only units from correct trials, of good quality, and within m2
units = myexp.select_units(group="good", name="m2", min_depth=200, max_depth=1200)

# Now align these firing rates to the point at which the trial began
duration = 4  # we want the 3s after trial began to capture the whole range.

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

firing_rates = myexp.align_trials(
    ActionLabels.clean, Events.reach_onset, "spike_rate", duration=duration, units=units
)
# From this, must now calculate the point at which the LED turned off in each trial.
# Will allow plotting of off time.
off_times = event_times("led_off", myexp)  # time after LED on when trial finished

# %%
# Now shall define a function to calculate per Trial FF


def cross_trial_FF(spike_times, exp, duration, bin_size, mean_matched=False):
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

    mean_matched: Whether to apply mean matching adjustment (as defined by Churchland 2010) to correct for large changes in unit mean firing rates across trials
                  If this is False, the returned dataframe will be of unadjusted single unit FFs


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
        session_FF = pd.DataFrame()

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
            FanoFactor_scores = {}
            for b in unit_bin_means.columns:

                # Isolate bin values, drop any units that are nonresponsive
                bin_means = unit_bin_means[b]
                bin_means = pd.Series.to_numpy(bin_means).reshape((-1, 1))
                bin_var = unit_bin_var[b].values

                # Construct linear regression with x axis as mean count, and y axis as variance
                raw_FF_model = LinearRegression()
                raw_FF_model.fit(bin_means, bin_var)

                # TODO: INSERT MEAN MATCHING STAGE

                # plot the slope for each bin as a point on a scatter
                FanoFactor_scores.update(
                    {b: raw_FF_model.coef_}
                )  # give the coefficient of the model, representing the slope of the line

            # update session scores
            FanoFactors = pd.DataFrame.from_dict(FanoFactor_scores)
            FanoFactors.index = [name]
            FanoFactors = FanoFactors.rename_axis(columns="bin", index="session")

            all_FF = pd.concat([all_FF, FanoFactors], axis=0)

        elif mean_matched is False:
            # Remember, this will return the individual FF for each unit, not the population level FF
            # calculate raw FF for each unit
            FanoFactors = pd.DataFrame(columns=["bin", "FF", "session", "unit"])
            for unit in unit_bin_means.index:
                unit_FF = pd.DataFrame(
                    unit_bin_var.loc[unit] / unit_bin_means.loc[unit]
                ).rename(
                    columns={unit: "FF"}
                )  # return all bins for this unit
                unit_FF = unit_FF.reset_index()
                unit_FF["session"] = name
                unit_FF["unit"] = unit

                FanoFactors = pd.concat([FanoFactors, unit_FF])

            # Reorder columns, then return data
            FanoFactors = FanoFactors[["session", "unit", "bin", "FF"]]
            FanoFactors.reset_index(drop=True)
            all_FF = pd.concat([all_FF, FanoFactors])

    ioutils.write_hdf5("/data/aidan/FF_Calculation.h5", all_FF)
    return all_FF


per_unit_FF = cross_trial_FF(
    spike_times=spike_times, exp=myexp, duration=4, bin_size=0.1, mean_matched=False
)


# %%
# Plot individual unit FFs for each session overlaid on firing rates

for s, session in enumerate(myexp):
    name = session.name
    units = per_unit_FF["unit"].unique()
    subplots = Subplots2D(units, sharex=False)

    palette = sns.color_palette()
    plt.rcParams["figure.figsize"] = (10, 10)
    ses_fr = firing_rates[s]

    for i, un in enumerate(units):
        unit_data = per_unit_FF.loc[per_unit_FF["unit"] == un]
        unit_data["bin/10"] = unit_data["bin"] / 10 - (
            duration / 2
        )  # convert 100ms bins to seconds relative to action
        ax = subplots.axes_flat[i]
        ax1 = ax.twinx()  # secondary axis for fr

        bins = np.unique(per_unit_FF.loc[per_unit_FF["session"] == name]["bin"].values)

        val_data = ses_fr[un].stack().reset_index()
        val_data["y"] = val_data[0]
        num_samples = len(ses_fr[values[0]].columns)

        p = sns.lineplot(
            data=unit_data, x="bin/10", y="FF", ax=ax, linewidth=0.5, color="red"
        )
        p.axvline(
            len(np.unique(bins)) / 2,
            color="green",
            ls="--",
        )
        # p.autoscale(enable=True, tight=False)
        p.set_xticks([])
        p.set_yticks([])
        p.get_yaxis().get_label().set_visible(False)
        p.get_xaxis().get_label().set_visible(False)
        ax.set_xlim(bins[0], bins[-1])

        q = sns.lineplot(
            data=val_data,
            x="time",
            y="y",
            ci="sd",
            ax=ax1,
            linewidth=0.5,
        )
        # q.autoscale(enable=True, tight=False)
        q.set_yticks([])
        q.set_xticks([])
        ax1.set_xlim(-duration / 2, duration / 2)
        peak = ses_fr[value].values.mean(axis=1).max()
        q.get_yaxis().get_label().set_visible(False)
        q.get_xaxis().get_label().set_visible(False)
        q.axvline(c=palette[1], ls="--", linewidth=0.5)
        q.set_box_aspect(1)

    to_label = subplots.axes_flat[0]
    to_label.get_yaxis().get_label().set_visible(True)
    to_label.get_xaxis().get_label().set_visible(True)
    to_label.set_xticks([])

    legend = subplots.legend

    legend.text(
        0,
        0.3,
        "Unit ID",
        transform=legend.transAxes,
        color=palette[0],
    )
    legend.set_visible(True)
    ticks = list(unit_data["bin"].unique())
    legend.get_xaxis().set_ticks(
        ticks=[ticks[0], ticks[round(len(ticks) / 2)], ticks[-1]]
    )
    legend.set_xlabel("100ms Bin")
    legend.get_yaxis().set_visible(False)
    legend.set_box_aspect(1)

    # Overlay firing rates
    values = ses_fr.columns.get_level_values("unit").unique()
    values.values.sort()
    num_samples = len(ses_fr[values[0]].columns)

    plt.suptitle(f"Session {name} Per Unit Fano Factor + Firing Rate")

    utils.save(f"/home/s1735718/Figures/{name}_per_unit_FanoFactor", nosize=True)
# %%
