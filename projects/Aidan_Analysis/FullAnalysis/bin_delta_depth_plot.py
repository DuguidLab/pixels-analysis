###Using the calculated bin changes from the FR anova, this script will plot the depths of individual neuronal units against the change in firing rate (delta)
# Delta calculated from start of drop in FR to end (i.e., largest change in bin)

#%%
# First import required packages:
import enum
from xml.etree.ElementInclude import include
from matplotlib.pyplot import legend, title, ylabel, ylim
import matplotlib.lines as mlines
from pyrsistent import b
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from base import *
from channeldepth import meta_spikeglx
from tqdm import tqdm  # for progress bar

from functions import (
    per_trial_binning,
    bin_data_average,
    event_times,
    per_trial_spike_rate,
)
from reach import Cohort


from pixtools.utils import Subplots2D
from pixtools import utils
from pixtools import spike_rate
from pixels import ioutils
import statsmodels.stats.multicomp as mc
from statsmodels.graphics.factorplots import interaction_plot
from bioinfokit.analys import stat
from channeldepth import meta_spikeglx
from pixtools import clusters
from textwrap import wrap

#%%
# Select units from given session
units = myexp.select_units(group="good", name="m2", min_depth=200, max_depth=1200)

duration = 2
per_unit_aligned = myexp.align_trials(
    ActionLabels.correct, Events.led_off, "spike_rate", duration=duration, units=units
)

#%%
# Using import functions, bin firing rate data
bin_data = per_trial_binning(per_unit_aligned, myexp, timespan=2, bin_size=0.1)
bin_data.reorder_levels(["session", "unit", "trial"], axis=1)

multicomp_bin_results = []
multicomp_unit_results = []
multicomp_int_results = []


#
###Run linear model in a distributed manner###
###NB: As the dataset is too large to run within memory constraints, will remove the final 4s of data.


def within_unit_GLM(bin_data, myexp):

    """
    Function takes binned data (with levels in order session -> unit -> trial) and performs an ANOVA on each session
    Returns the multiple comparisons and ANOVA results for the interaction, also prints anova results
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

    # Now split this data into sessions
    results = []
    for s, session in enumerate(myexp):

        name = session.name
        ses_data = bin_data[s]

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
        results.append(output)

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
    results = pd.DataFrame(results)
    return multicomp_int_results, results


glm, output = within_unit_GLM(bin_data, myexp)

#Cache the multicomps and results
for s, session in enumerate(myexp):
    name = session.name
    ses_output = output[s]
    ses_output = pd.DataFrame(ses_output)
    ioutils.write_hdf5(f"/data/aidan/glm_output_session_{name}.h5", ses_output)

    ses_multicomps = glm[s]
    ses_multicomps = pd.DataFrame(ses_multicomps)

    ioutils.write_hdf5(f"/data/aidan/glm_multicomps_session_{name}.h5", ses_multicomps)

#For later retrieval
# %%
# Using this data, now shall determine the largest delta between bins for each unit
# Split data into seperate dataframes for each session
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


sig_deltas = unit_delta(
    glm, myexp, bin_data, bin_duration=0.1, sig_only=True, percentage_change=False
)
all_deltas = unit_delta(
    glm, myexp, bin_data, bin_duration=0.1, sig_only=False, percentage_change=False
)

# %%
# Using this delta information, the approximate depths of these units may be determined
# First, update the unit depths function
def unit_depths(exp):
    """
    Parameters
    ==========
    exp : pixels.Experiment
        Your experiment.
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


all_depths = unit_depths(myexp)
# %%
# Sort pyramidal and interneuronal units for the plot below
# select all pyramidal neurons
pyramidal_units = get_pyramidals(myexp)

# and all interneurons
interneuron_units = get_interneurons(myexp)

pyramidal_aligned = myexp.align_trials(
    ActionLabels.correct,
    Events.led_off,
    "spike_rate",
    duration=2,
    units=pyramidal_units,
)

interneuron_aligned = myexp.align_trials(
    ActionLabels.correct,
    Events.led_off,
    "spike_rate",
    duration=2,
    units=interneuron_units,
)


#%%
# Now extract the depths of each of our largest delta units from this
# Plot the points as interneurons or pyramidals

plt.rcParams.update({"font.size": 20})
for s, session in enumerate(myexp):
    name = session.name
    ses_deltas = sig_deltas[s]

    ses_deltas = pd.DataFrame(ses_deltas, columns=["unit", "delta"])

    ses_depths = all_depths[name]
    # Find depths of associated units
    unit_depth = ses_depths[ses_deltas["unit"]].melt()
    ses_deltas = pd.concat([ses_deltas, unit_depth["value"]], axis=1)
    ses_deltas.rename(columns={"value": "depth"}, inplace=True)

    # Do the same for nonsignificant units
    ses_nosig_deltas = all_deltas[s]
    ses_nosig_deltas = pd.DataFrame(ses_nosig_deltas, columns=["unit", "delta"])

    # Find depths of associated units
    unit_nosig_depth = ses_depths[ses_nosig_deltas["unit"]].melt()
    ses_nosig_deltas = pd.concat([ses_nosig_deltas, unit_nosig_depth["value"]], axis=1)
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
    for i, item in enumerate(ses_deltas["unit"]):
        if item in pyr_units:
            unit_types.append("pyramidal")
        if item in int_units:
            unit_types.append("interneuron")
        else:
            continue

    nosig_unit_types = []
    for i, item in enumerate(ses_nosig_deltas["unit"]):
        if item in pyr_units:
            nosig_unit_types.append("pyramidal")
        if item in int_units:
            nosig_unit_types.append("interneuron")
        else:
            continue

    # Append unit_types as a new column
    # Will try and append this as a new column, to see where the extra value lies
    unit_types = pd.DataFrame(unit_types)
    ses_deltas = pd.concat([ses_deltas, unit_types], ignore_index=True, axis=1)
    ses_deltas.rename(
        columns={0: "unit", 1: "delta", 2: "depth", 3: "type"}, inplace=True
    )

    nosig_unit_types = pd.DataFrame(nosig_unit_types)
    ses_nosig_deltas = pd.concat(
        [ses_nosig_deltas, nosig_unit_types], ignore_index=True, axis=1
    )
    ses_nosig_deltas.rename(
        columns={0: "unit", 1: "delta", 2: "depth", 3: "type"}, inplace=True
    )

    # Finally, plot a graph of deltas relative to zero, by depth
    p = sns.scatterplot(
        x="delta",
        y="depth",
        data=ses_nosig_deltas,
        s=100,
        linewidth=1,
        style="type",
        color="#972c7f",
        alpha=0.5,
    )

    p2 = sns.scatterplot(
        x="delta",
        y="depth",
        data=ses_deltas,
        s=100,
        linewidth=1,
        style="type",  # uncomment when neuron type is fixed
        color="#221150",
    )

    plt.axvline(x=0, color="green", ls="--")

    plt.xlim(-50, 50)
    plt.ylim(0, 1300)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("Δ Firing Rate (Hz)", fontsize=20)
    plt.ylabel("Depth of Recorded Neuron (μm)", fontsize=20)
    plt.suptitle(
        "\n".join(
            wrap(
                f"Largest Trial Firing Rate Change by pM2 Depth - Session {name}",
                width=30,
            )
        ),
        y=1.1,
        fontsize=20,
    )

    # Create lengend manually
    # Entries for legend listed below as handles:
    sig_dot = mlines.Line2D(
        [],
        [],
        color="#221150",
        marker="o",
        linestyle="None",
        markersize=10,
        label="Significant",
    )
    nonsig_dot = mlines.Line2D(
        [],
        [],
        color="#972c7f",
        marker="o",
        linestyle="None",
        alpha=0.5,
        markersize=10,
        label="Non-Significant",
    )
    grey_dot = mlines.Line2D(
        [],
        [],
        color="grey",
        marker="o",
        linestyle="None",
        markersize=10,
        label="Pyramidal",
    )
    grey_cross = mlines.Line2D(
        [],
        [],
        color="grey",
        marker="X",
        lw=5,
        linestyle="None",
        markersize=10,
        label="Interneuronal",
    )

    plt.legend(
        handles=[sig_dot, nonsig_dot, grey_dot, grey_cross],
        bbox_to_anchor=(1.7, 1),
        title="Significance (p < 0.05)",
        fontsize=15,
    )

    # Now invert y-axis and save
    plt.gca().invert_yaxis()

    utils.save(
        f"/home/s1735718/Figures/{name}_DeltaFR_byDepth",
        nosize=True,
    )
    plt.show()

# %%
