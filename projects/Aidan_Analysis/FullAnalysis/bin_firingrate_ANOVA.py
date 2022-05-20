#################################################
# This script will analyse the standard deviation of the data, as a measure of variance
# Rather than confidence interval size.
# Will be done by a comparison across bins and session
# IV: Standard Deviation of Firing Rate
# DV: Bin (Within Subjects), Session (Between Subjects)
#################################################

# Import req. packages

import enum
from xml.etree.ElementInclude import include
from matplotlib.pyplot import title, ylabel, ylim
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from base import *


from functions import (
    per_trial_binning,
    bin_data_average,
    event_times,
    per_trial_spike_rate,
)
from reach import Cohort


sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")
sys.path.insert(0, "/home/s1735718/PixelsAnalysis/pixels/pixels")
from pixtools.utils import Subplots2D  # use the local copy of base.py
from pixtools import utils
from pixtools import spike_rate
from pixels import ioutils
import statsmodels.stats.multicomp as mc
from statsmodels.graphics.factorplots import interaction_plot
from bioinfokit.analys import stat

#%%
# Select Units From Cache (m2 only)
units = myexp.select_units(group="good", name="m2", min_depth=200, max_depth=1200)

duration = 2
trial_aligned = myexp.align_trials(
    ActionLabels.correct, Events.led_off, "spike_rate", duration=duration, units=units
)

#%%
# Bin firing rate data
#
bin_data = per_trial_binning(trial_aligned, myexp, timespan=2, bin_size=0.1)


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


bin_sd = bin_data_SD(bin_data)


# %%
# Now take SD data and and convert to long form to prepare for ANOVA analysis
long_sd = pd.melt(
    bin_sd.reset_index(), id_vars=["session", "trial"]
)  # Preserve the indexes as comparator variables
long_sd.rename(
    {"value": "firing_rate_sd"}, axis=1, inplace=True
)  # rename value column to sd for readability

#%%
#######################################
# Then shall run a mixed factorial ANOVA
# IV: Standard Deviation of Firing Rate
# DV: Bin (Within Subjects - 100 levels), Session (Between Subjects - 2 levels for VR46 [n otherwise])
#######################################

# Assemble model
model = ols(
    "firing_rate_sd ~ C(bin) + C(session) + C(bin):C(session)", data=long_sd
).fit()

output = sm.stats.anova_lm(
    model, typ=2
)  # Run test on model, sample sizes are equal between groups here (neurons stay the same across trials) so type 2 SS used
print(output)  # Print output

####Run multiple comparisons####
# For main effect of bin
res = stat()
res.tukey_hsd(
    df=long_sd,
    res_var="firing_rate_sd",
    xfac_var="bin",
    anova_model="firing_rate_sd ~ C(bin) + C(session) + C(bin):C(session)",
    phalpha=0.05,
    ss_typ=2,
)
bin_main_effect = res.tukey_summary

# For main effect of session
res = stat()
res.tukey_hsd(
    df=long_sd,
    res_var="firing_rate_sd",
    xfac_var="session",
    anova_model="firing_rate_sd ~ C(bin) + C(session) + C(bin):C(session)",
    phalpha=0.05,
    ss_typ=2,
)
ses_main_effect = res.tukey_summary

# For interaction
res = stat()
res.tukey_hsd(
    df=long_sd,
    res_var="firing_rate_sd",
    xfac_var=["bin", "session"],
    anova_model="firing_rate_sd ~ C(bin) + C(session) + C(bin):C(session)",
    phalpha=0.05,
    ss_typ=2,
)
interaction = res.tukey_summary

# Determine sig. comparisons
sig_bin_comps = bin_main_effect.loc[bin_main_effect["p-value"] < 0.05]
sig_ses_comps = ses_main_effect.loc[ses_main_effect["p-value"] < 0.05]
sig_int = interaction.loc[interaction["p-value"] < 0.05]

####Now plot the findings of this analysis####
fig = interaction_plot(
    x=long_sd["bin"], trace=long_sd["session"], response=long_sd["firing_rate_sd"]
)
plt.show()
# From this it is clear there is a drop around the centre of bins

##Check assumptions
# First residual plot (QQ)
sm.qqplot(res.anova_std_residuals, line="45")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
# plt.show()

# Then Distribution plot
plt.hist(res.anova_model_out.resid, bins="auto", histtype="bar", ec="k")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
# plt.show()

import scipy.stats as stats

w, pvalue = stats.shapiro(res.anova_model_out.resid)
# print(w, pvalue)

# Model assumptions are met.
# %%
#############################################
# Run the same analysis on firing rate to paint the other half of the picture.
# Calculate mean firing rate for the data
bin_avgs = bin_data_average(bin_data)

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

long_mean = pd.melt(
    means.reset_index(), id_vars=["session", "trial"]
)  # Preserve the indexes as comparator variables
long_mean.rename(
    {"value": "mean_firing_rate"}, axis=1, inplace=True
)  # rename value column to sd for readability

#%%
# Now run analysis for firing rate, per session
# Do this manually, its faster

model = ols(
    "mean_firing_rate ~ C(bin) + C(session) + C(bin):C(session)", data=long_mean
).fit()

output = sm.stats.anova_lm(
    model, typ=2
)  # Run test on model, sample sizes are equal between groups here (neurons stay the same across trials) so type 2 SS used
print(output)  # Print output

####Run multiple comparisons####
# For main effect of bin
res = stat()
res.tukey_hsd(
    df=long_mean,
    res_var="mean_firing_rate",
    xfac_var="bin",
    anova_model="mean_firing_rate ~ C(bin) + C(session) + C(bin):C(session)",
    phalpha=0.05,
    ss_typ=2,
)
bin_main_effect_fr = res.tukey_summary

# For main effect of session
res = stat()
res.tukey_hsd(
    df=long_mean,
    res_var="mean_firing_rate",
    xfac_var="session",
    anova_model="mean_firing_rate ~ C(bin) + C(session) + C(bin):C(session)",
    phalpha=0.05,
    ss_typ=2,
)
ses_main_effect_fr = res.tukey_summary

# For interaction
res = stat()
res.tukey_hsd(
    df=long_mean,
    res_var="mean_firing_rate",
    xfac_var=["bin", "session"],
    anova_model="mean_firing_rate ~ C(bin) + C(session) + C(bin):C(session)",
    phalpha=0.05,
    ss_typ=2,
)
interaction_fr = res.tukey_summary

# Determine sig. comparisons
sig_bin_comps_fr = bin_main_effect_fr.loc[bin_main_effect_fr["p-value"] < 0.05]
sig_ses_comps_fr = ses_main_effect_fr.loc[ses_main_effect_fr["p-value"] < 0.05]
sig_int_fr = interaction_fr.loc[interaction_fr["p-value"] < 0.05]

# %%
######Now determine where sig. changes occur in adjacent bins######
##To do so, will only select columns 1s before grasp.
def pairwise_significance_extraction(pairwise_data):

    """
    This function will return pairwise comparisons for the 1s preceeding grasp.
    To return only significant values, feed it data with sig. isolated already (i.e., where p-value column < 0.05)

    pairwise_data: the result of the multiple comparisons performed earlier in the script, by tukey_hsd()

    """

    df = pairwise_data.loc[
        (
            (pairwise_data["group1"] == "-1.0, -0.9")
            | (pairwise_data["group1"] == "-0.9, -0.8")
            | (pairwise_data["group1"] == "-0.8, -0.7")
            | (pairwise_data["group1"] == "-0.7, -0.6")
            | (pairwise_data["group1"] == "-0.6, -0.5")
            | (pairwise_data["group1"] == "-0.5, -0.4")
            | (pairwise_data["group1"] == "-0.4, -0.3")
            | (pairwise_data["group1"] == "-0.3, -0.2")
            | (pairwise_data["group1"] == "-0.2, -0.1")
            | (pairwise_data["group1"] == "-0.1, 0.0005")
        )
        & (
            (pairwise_data["group2"] == "-1.0, -0.9")
            | (pairwise_data["group2"] == "-0.9, -0.8")
            | (pairwise_data["group2"] == "-0.8, -0.7")
            | (pairwise_data["group2"] == "-0.7, -0.6")
            | (pairwise_data["group2"] == "-0.6, -0.5")
            | (pairwise_data["group2"] == "-0.5, -0.4")
            | (pairwise_data["group2"] == "-0.4, -0.3")
            | (pairwise_data["group2"] == "-0.3, -0.2")
            | (pairwise_data["group2"] == "-0.2, -0.1")
            | (pairwise_data["group2"] == "-0.1, 0.0005")
        )
    ]

    return df


# Return only comparisons in the 1s leading up to grasp
# In SD data, significant change occurs in the 0.6s leading up to grasp
bin_sd_comps = pairwise_significance_extraction(sig_bin_comps)
bin_fr_comps = pairwise_significance_extraction(sig_bin_comps_fr)
######################################
# TODO:
# Run this on remainder of sessions
# run on all session in one Cohort
# plot significant changes to graph
######################################
