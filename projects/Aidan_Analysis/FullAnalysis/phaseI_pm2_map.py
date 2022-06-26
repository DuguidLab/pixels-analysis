"""
This file contains the first section of my full analysis, it will aim to do the following:

1. Determine if the distribution of pyramidal to interneuron type alignes with known data. This will give an idea of the rate of undersampling by the probe
2. Concatenate as much information surrounding unit activity as possible. i.e., type, activity, statistically significant changes, depth etc. 
   This will be concatenated and saved as a dataframe with the following structure:

   ---------------------------------------------------------------------------------
   |Session | Unit No | Delta Change | Depth (um) | Supra/Infragranular | Neuron Type| Delta Significance|
   ---------------------------------------------------------------------------------
"""

# First import required packages
from argparse import Action
from base import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sc
import sys
from channeldepth import meta_spikeglx
import statsmodels.api as sm
from statsmodels.formula.api import ols
from reach import Cohort
from xml.etree.ElementInclude import include
from matplotlib.pyplot import legend, title, ylabel, ylim
import matplotlib.lines as mlines
from tqdm import tqdm
from pixels import ioutils

from functions import (
    event_times,
    unit_depths,
    per_trial_binning,
    event_times,
    within_unit_GLM,
    bin_data_average,
    # unit_delta, #for now use a local copy
)

#%%
# First select units based on spike width analysis
all_units = myexp.select_units(group="good", name="m2", min_depth=200, max_depth=1200)

pyramidals = get_pyramidals(myexp)
interneurons = get_interneurons(myexp)


# Then import the known neuronal type distributions from csv
known_units = pd.read_csv("cell_atlas_neuron_distribution_M2.csv", index_col=False)

#%%
# Now align Trials
per_unit_aligned = myexp.align_trials(
    ActionLabels.correct, Events.led_off, "spike_rate", duration=2, units=all_units
)

left_aligned = myexp.align_trials(
    ActionLabels.correct_left, Events.led_off, "spike_rate", duration=2, units=all_units
)

right_aligned = myexp.align_trials(
    ActionLabels.correct_right,
    Events.led_off,
    "spike_rate",
    duration=2,
    units=all_units,
)

pyramidal_aligned = myexp.align_trials(
    ActionLabels.correct, Events.led_off, "spike_rate", duration=2, units=pyramidals
)
interneuron_aligned = myexp.align_trials(
    ActionLabels.correct, Events.led_off, "spike_rate", duration=2, units=interneurons
)

#%%
##Now that we have determined the optimal sessions to take from each mouse
# Create a dataframe of neuron type and depth

# First bin data, then shift hierachy
bin_data = per_trial_binning(per_unit_aligned, myexp, timespan=2, bin_size=0.1)
bin_data.reorder_levels(["session", "unit", "trial"], axis=1)

# Then perform an ANOVA
glm = within_unit_GLM(bin_data, myexp)

# Finally take average firing rate
fr_avgs = per_unit_aligned.mean()
fr_avgs = fr_avgs.groupby(["session", "unit"]).mean()

#%%
# This cell is to serve as an interactive method of altering the deltas function
# The aim is to allow selection of the type of delta
# I.e., first, or largest


def unit_delta(glm, myexp, bin_data, sig_only, delta_type, percentage_change):
    """
    Function takes the output of the multiple comparisons calculated by within_unit_GLM() and calculates the change in firing rate (or other IV measured by the GLM) between bins, per unit
    This requires both experimental cohort data, and bin_data calculated by per_trial_binning() and passed through reorder levels (as "session", "unit", "trial")
    Function returns a list of lists, in a hierarchy of [session[unit deltas]]

    Using this data, a depth plot may be created displaying the relative delta by unit location in pM2.

    glm: the result of the ANOVA and subsequent Tukey adjusted multiple comparisons performed by within_unit_GLM
         NB: THIS IS A COMPUTATIONALLY EXPENSIVE FUNCTION, DO NOT RUN IT ON MORE THAN ONE SESSION AT A TIME

    myexp: the experimental cohort defined in base.py

    bin_data: the binned raw data computed by the per_trial_binning function

    delta_type: the type of delta change to return, i.e., "largest", "first", or "last"

    sig_only: whether to return only the greatest delta of significant bins or not

    percentage_change: whether to return deltas as a percentage change

    """
    ses_deltas = []
    print("beginning unit delta calculation")
    for s, session in enumerate(myexp):

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

            # Now determine which delta to obtain
            try:
                delta_type == "largest" or delta_type == "first" or delta_type == "last"

                if delta_type == "largest":

                    # find row with largest difference
                    if unit_comps.empty:  # skip if empty
                        continue

                    row = unit_comps["Diff"].idxmax()
                    sigunit_comps.append(unit_comps[unit_comps.index == row])

                if delta_type == "first":

                    if unit_comps.empty:
                        continue

                    row = unit_comps[
                        unit_comps.index == unit_comps.index[0]
                    ]  # get the first delta that occurs
                    sigunit_comps.append(row)  # append this to the list

                if delta_type == "last":

                    if unit_comps.empty:
                        continue

                    row = unit_comps[unit_comps.index == unit_comps.index[-1]]

            except:
                print("Invalid Delta_Type Specified")
                break

        # Now that we know the units with significant comparisons, take these bins from raw firing rate averages
        ses_avgs = bin_data_average(bin_data[s])

        # Iterate through our significant comparisons, calculating the actual delta firing rate
        for i in range(len(sigunit_comps)):
            sig_comp = sigunit_comps[i]

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
                print(delta)
                unit_deltas.append(
                    [
                        int(unit[0]),
                        delta,
                        ses_unit[bin_val2],
                        ses_unit[bin_val1],
                        bin_val2,
                        bin_val1,
                    ]
                )
            # Finally, get the delta value across these bins for the given unit
            elif percentage_change == False:
                delta = ses_unit[bin_val2] - ses_unit[bin_val1]
                unit_deltas.append(
                    [
                        int(unit[0]),
                        delta,
                        ses_unit[bin_val2],
                        ses_unit[bin_val1],
                        bin_val2,
                        bin_val1,
                    ]
                )

        ses_deltas.append(unit_deltas)

    return ses_deltas


largest_sig_deltas = unit_delta(
    glm, myexp, bin_data, sig_only=True, delta_type="largest", percentage_change=False
)

deltas = unit_delta(
    glm, myexp, bin_data, sig_only=False, delta_type="largest", percentage_change=False
)
#%%
# Now concatenate this information into a single dataframe
# TODO: Add the ability to extract the bin beginnings and ends including firing rates to this function.

depths = unit_depths(myexp)
layer_info = pd.read_csv("granulardepths_VR46_VR59.csv")


def unit_info_concatenation(
    average_fr,
    sig_deltas,
    all_deltas,
    depths,
    layer_info,
    pyramidal_aligned,
    interneuron_aligned,
    myexp,
):

    """
    This function takes all previously calculated information surrounding unit activity, type, depth, etc. and concatenates it as a single dataframe
    Requires all of these values are previously determined using predefined functions shown

    average_fr: Average firing rate for each unit, across trials, organised in long form as a multiindexed dataframe (session->unit)

    sig_deltas: Significant delta changes to firing rate (or SD) determined by within_unit_glm

    all_deltas: All delta changes, regardless of significance, determined by the same within_unit_glm function

    depths: Depths of all units across sessions, determined by unit_depths

    layer_info: Dataframe containing the granular layer boundaries defined through histological analyses

    pyramidal_aligned: Only pyramidal units, aligned to the same event, determined by myexp.align_trials

    interneuron_aligned: Only interneuronal units, aligned to the same event

    myexp: The experimental cohort defined in base.py

    """
    unit_info = pd.DataFrame()
    for s, session in enumerate(myexp):
        name = session.name
        ses_deltas = sig_deltas[s]

        ses_deltas = pd.DataFrame(
            ses_deltas,
            columns=[
                "unit",
                "delta",
                "start value",
                "end value",
                "start bin",
                "end bin",
            ],
        )
        ses_depths = depths[name]

        # Determine depths of units
        unit_depth = ses_depths[ses_deltas["unit"]].melt()
        ses_deltas = pd.concat([ses_deltas, unit_depth["value"]], axis=1)
        ses_deltas.rename(columns={"value": "depth"}, inplace=True)

        # Do the same for nonsignificant units
        ses_nosig_deltas = all_deltas[s]
        ses_nosig_deltas = pd.DataFrame(
            ses_nosig_deltas,
            columns=[
                "unit",
                "delta",
                "start value",
                "end value",
                "start bin",
                "end bin",
            ],
        )

        # Find depths of associated units
        unit_nosig_depth = ses_depths[ses_nosig_deltas["unit"]].melt()
        ses_nosig_deltas = pd.concat(
            [ses_nosig_deltas, unit_nosig_depth["value"]], axis=1
        )
        ses_nosig_deltas.rename(columns={"value": "depth"}, inplace=True)
        # Now remove any significant values
        ses_nosig_deltas = ses_nosig_deltas[
            ~ses_nosig_deltas["unit"].isin(ses_deltas["unit"])
        ].dropna()
        ses_nosig_deltas = ses_nosig_deltas.reset_index(drop=True)

        ##Then determine the type of neuron these units are (i.e., pyramidal or interneuron)
        pyr_units = pyramidal_aligned[s].melt()["unit"].unique()
        int_units = interneuron_aligned[s].melt()["unit"].unique()
        dicts = {}

        unit_types = []
        for item_sig in ses_deltas["unit"]:
            if item_sig in pyr_units:
                unit_types.append("pyramidal")
            if item_sig in int_units:
                unit_types.append("interneuron")
            if item_sig not in pyr_units and item_sig not in int_units:
                unit_types.append("neuron unclassified")

        nosig_unit_types = []
        for item in ses_nosig_deltas["unit"]:
            if item in pyr_units:
                nosig_unit_types.append("pyramidal")
            if item in int_units:
                nosig_unit_types.append("interneuron")
            if item not in pyr_units and item not in int_units:
                nosig_unit_types.append("neuron unclassified")

        # Append unit_types as a new column
        # Will try and append this as a new column, to see where the extra value lies
        unit_types = pd.DataFrame(unit_types)
        ses_deltas = pd.concat([ses_deltas, unit_types], ignore_index=True, axis=1)
        ses_deltas.rename(
            columns={
                0: "unit",
                1: "bin1 val",
                2: "bin2 val",
                3: "delta",
                4: "start bin",
                5: "end bin",
                6: "depth",
                7: "type",
            },
            inplace=True,
        )
        ses_deltas["delta significance"] = "significant"
        ses_deltas.insert(0, "session", "")
        ses_deltas["session"] = name

        nosig_unit_types = pd.DataFrame(nosig_unit_types)
        ses_nosig_deltas = pd.concat(
            [ses_nosig_deltas, nosig_unit_types], ignore_index=True, axis=1
        )
        ses_nosig_deltas.rename(
            columns={
                0: "unit",
                1: "bin1 val",
                2: "bin2 val",
                3: "delta",
                4: "start bin",
                5: "end bin",
                6: "depth",
                7: "type",
            },
            inplace=True,
        )
        ses_nosig_deltas["delta significance"] = "non-significant"
        ses_nosig_deltas.insert(0, "session", "")
        ses_nosig_deltas["session"] = name
        # Concatenate sig and nonsig deltas as unit types
        session_info = pd.concat(
            [ses_deltas, ses_nosig_deltas], ignore_index=True, axis=0
        )

        # Now iterate through session info by row, classifying depth into supra or infragranular
        # Will use the layer_info to do so
        mouse_id = session.name[-4:]
        mouse_boundary = (
            layer_info.loc[layer_info["MouseID"] == mouse_id]["LayerV_Depth"].item()
            * 1000
        )  # get value and convert from mm to um

        unit_depth_class = []
        for index, row in session_info.iterrows():
            if row["depth"] < mouse_boundary:
                unit_depth_class.append("Supragranular")

            if row["depth"] > mouse_boundary:
                unit_depth_class.append("Infragranular")

        # Append this to the session_info as a new column (after depth)
        session_info.insert(4, "Depth Classification", unit_depth_class)

        # Now add firing rates
        ses_rates = average_fr[s].reindex(
            index=session_info["unit"]
        )  # reorder this to match out data
        session_info.insert(
            len(session_info.columns), "average firing rate", ses_rates.values
        )

        # Finally, concatenate this info for the session into the master dataframe to be returned
        unit_info = pd.concat([unit_info, session_info], axis=0, ignore_index=True)

    return unit_info


data = unit_info_concatenation(
    fr_avgs,
    largest_sig_deltas,
    deltas,
    depths,
    layer_info,
    pyramidal_aligned=pyramidal_aligned,
    interneuron_aligned=interneuron_aligned,
    myexp=myexp,
)

# %%
# Now finally, will export this data as a saved file using utils.save
ioutils.write_hdf5("/data/aidan/concatenated_session_information.h5", data)


# %%


# def tester(
#     average_fr,
#     sig_deltas,
#     all_deltas,
#     depths,
#     layer_info,
#     myexp,
# ):

#     """
#     This function takes all previously calculated information surrounding unit activity, type, depth, etc. and concatenates it as a single dataframe
#     Requires all of these values are previously determined using predefined functions shown

#     average_fr: Average firing rate for each unit, across trials, organised in long form as a multiindexed dataframe (session->unit)

#     sig_deltas: Significant delta changes to firing rate (or SD) determined by within_unit_glm

#     all_deltas: All delta changes, regardless of significance, determined by the same within_unit_glm function

#     depths: Depths of all units across sessions, determined by unit_depths

#     layer_info: Dataframe containing the granular layer boundaries defined through histological analyses

#     pyramidal_aligned: Only pyramidal units, aligned to the same event, determined by myexp.align_trials

#     interneuron_aligned: Only interneuronal units, aligned to the same event

#     myexp: The experimental cohort defined in base.py

#     """
#     unit_info = pd.DataFrame()
#     for s, session in enumerate(myexp):
#         name = session.name
#         ses_deltas = sig_deltas[s]

#         ses_deltas = pd.DataFrame(ses_deltas, columns=["unit", "delta"])
#         ses_depths = depths[name]

#         # Determine depths of units
#         unit_depth = ses_depths[ses_deltas["unit"]].melt()
#         ses_deltas = pd.concat([ses_deltas, unit_depth["value"]], axis=1)
#         ses_deltas.rename(columns={"value": "depth"}, inplace=True)

#         # Do the same for nonsignificant units
#         ses_nosig_deltas = all_deltas[s]
#         ses_nosig_deltas = pd.DataFrame(ses_nosig_deltas, columns=["unit", "delta"])

#         # Find depths of associated units
#         unit_nosig_depth = ses_depths[ses_nosig_deltas["unit"]].melt()
#         ses_nosig_deltas = pd.concat(
#             [ses_nosig_deltas, unit_nosig_depth["value"]], axis=1
#         )
#         ses_nosig_deltas.rename(columns={"value": "depth"}, inplace=True)
#         # Now remove any significant values
#         ses_nosig_deltas = ses_nosig_deltas[
#             ~ses_nosig_deltas["unit"].isin(ses_deltas["unit"])
#         ].dropna()
#         ses_nosig_deltas = ses_nosig_deltas.reset_index(drop=True)

#         # Append unit_types as a new column
#         # Will try and append this as a new column, to see where the extra value lies
#         unit_types = pd.DataFrame(unit_types)
#         ses_deltas.rename(columns={0: "unit", 1: "delta", 2: "depth"}, inplace=True)
#         ses_deltas["delta significance"] = "significant"
#         ses_deltas.insert(0, "session", "")
#         ses_deltas["session"] = name

#         nosig_unit_types = pd.DataFrame(nosig_unit_types)
#         ses_nosig_deltas.rename(
#             columns={0: "unit", 1: "delta", 2: "depth"}, inplace=True
#         )
#         ses_nosig_deltas["delta significance"] = "non-significant"
#         ses_nosig_deltas.insert(0, "session", "")
#         ses_nosig_deltas["session"] = name
#         # Concatenate sig and nonsig deltas as unit types
#         session_info = pd.concat(
#             [ses_deltas, ses_nosig_deltas], ignore_index=True, axis=0
#         )

#         # Now iterate through session info by row, classifying depth into supra or infragranular
#         # Will use the layer_info to do so
#         mouse_id = session.name[-4:]
#         mouse_boundary = (
#             layer_info.loc[layer_info["MouseID"] == mouse_id]["LayerV_Depth"].item()
#             * 1000
#         )  # get value and convert from mm to um

#         unit_depth_class = []
#         for index, row in session_info.iterrows():
#             if row["depth"] < mouse_boundary:
#                 unit_depth_class.append("Supragranular")

#             if row["depth"] > mouse_boundary:
#                 unit_depth_class.append("Infragranular")

#         # Append this to the session_info as a new column (after depth)
#         session_info.insert(4, "Depth Classification", unit_depth_class)

#         # Now add firing rates
#         ses_rates = average_fr[s].reindex(
#             index=session_info["unit"]
#         )  # reorder this to match out data
#         session_info.insert(
#             len(session_info.columns), "average firing rate", ses_rates.values
#         )

#         # Finally, concatenate this info for the session into the master dataframe to be returned
#         unit_info = pd.concat([unit_info, session_info], axis=0, ignore_index=True)

#     return unit_info


# test = tester(
#     fr_avgs,
#     largest_sig_deltas,
#     deltas,
#     depths,
#     layer_info,
#     myexp=myexp,
# )
ioutils.write_hdf5("/data/aidan/concatenated_session_information_test.h5", test)
# %%
