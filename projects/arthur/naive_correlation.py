"""
Pairwise unit correlation between PPC & M2.

ipsi/contra refers to anatomical relationship, left/right refers to visual stim.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.stats as sms
import seaborn as sns
import math

from scipy.stats import spearmanr
from statsmodels.stats import multitest

from pathlib import Path

import pandas as pd
import numpy as np

from pixels import Experiment, ioutils
from pixels.behaviours.reach import VisualOnly, ActionLabels, Events
from pixtools import spike_rate, utils, correlation

mice = [
    "HFR20",
    "HFR22",
    "HFR23",
]

exp = Experiment(
    mice,
    VisualOnly,
    "~/duguidlab/visuomotor_control/neuropixels",
    "~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON",
)

fig_dir = Path("~/duguidlab/visuomotor_control/AZ_notes/npx-plots/naive")
results_dir = Path("~/duguidlab/visuomotor_control/neuropixels/interim/results")
save_hdf5 = True

# do correlations on responsive units only?
do_resps_corr = False

# set subplots axes
cmap = sns.color_palette("icefire", as_cmap=True)

## Select units
print('get units...')
duration = 2
units = exp.select_units(
    min_depth=0,
    max_depth=1200,
    name="cortex0-1200",
)

# get spike rate for left & right visual stim.
print('get units spike rates...')
stim_left = exp.align_trials(
    ActionLabels.naive_left,
    Events.led_on,
    "spike_rate",
    units=units,
    duration=duration,
)
stim_right = exp.align_trials(
    ActionLabels.naive_right,
    Events.led_on,
    "spike_rate",
    units=units,
    duration=duration,
)

# name: 'side of visual stim'_'cc positivity'_'other relavant stuff'
left_pos_cc_max = []
left_neg_cc_max = []
right_pos_cc_max = []
right_neg_cc_max = []

left_pos_m2_counts = []
left_neg_m2_counts = []
right_pos_m2_counts = []
right_neg_m2_counts = []

left_pos_ppc_counts = []
left_neg_ppc_counts = []
right_pos_ppc_counts = []
right_neg_ppc_counts = []

for session in range(len(exp)):
    name = exp[session].name
    print(name)

    stims = [
        stim_left[session],
        stim_right[session],
    ]

    if do_resps_corr:

        # get responsive unit ids
        naive_m2_resps = ioutils.read_hdf5(
            results_dir / f"naive_m2_resps_units.h5"
        )
        naive_ppc_resps = ioutils.read_hdf5(
            results_dir / f"naive_ppc_resps_units.h5"
        )

        m2_units = naive_m2_resps[session].dropna()
        ppc_units = naive_ppc_resps[session].dropna()
        cache_file_p = exp[session].interim / "cache" / f"naive_resps_correlation_results_p_{duration}.npy"
        cache_file_cc = exp[session].interim / "cache" / f"naive_resps_correlation_results_cc_{duration}.npy"

    else:
        cache_file_p = exp[session].interim / "cache" / f"correlation_results_p_{duration}.npy"
        cache_file_cc = exp[session].interim / "cache" / f"correlation_results_cc_{duration}.npy"
        m2_units = units[session][0]
        ppc_units = units[session][1]

    if cache_file_p.exists() and cache_file_cc.exists():
        results_p = np.load(cache_file_p)
        results_cc = np.load(cache_file_cc)

    else:
        results_cc = np.zeros((len(m2_units), len(ppc_units), 2))
        results_p = np.zeros((len(m2_units), len(ppc_units), 2))

        """
        Correlate unit a from M2 with b from PPC, with all spike rate trace
        concatenated, i.e., 'reduce' the dimension (time). side0=left, side1=right.
        determine ipsi/contra manually. naive M2 recording are all on left.
        """
        for a, m2_unit in enumerate(m2_units):
            for s, side in enumerate(stims):
                a_trials = side[0][m2_unit]
                # reduce dimension
                a_trials = np.squeeze(a_trials.values.reshape((-1, 1)))

                for b, ppc_unit in enumerate(ppc_units):
                    b_trials = side[1][ppc_unit]
                    b_trials = np.squeeze(b_trials.values.reshape((-1, 1)))
                    cc, p = spearmanr(a_trials, b_trials)

                    # if most values are constant, Spearman r returns NaN.
                    # Replace NaN cc by 0, and NaN p-values by 1.
                    if math.isnan(cc):
                        print("nan CC:", m2_unit, ppc_unit)
                        cc = 0
                    if math.isnan(p):
                        print("nan p:", m2_unit, ppc_unit)
                        p = 1

                    results_cc[a, b, s] = cc
                    results_p[a, b, s] = p

            # correct p-values by FDR (false discovery rate), where FDR=FP/(FP+TP).
            # returned results_p is boolean, True means alpha<0.05.
        results_p_corrected, _ = sms.multitest.fdrcorrection(
            results_p.reshape((-1,)), alpha=0.05
        )
        results_p = results_p_corrected.reshape(results_p.shape)
        results_p = results_p.astype(int)
       
        np.save(cache_file_p, results_p)
        np.save(cache_file_cc, results_cc)
       
    # filter correlation coefficient matrix by its p-value matrix, thus only
    # those pairs that are sig. correlated are left.
    cc_sig = results_cc * results_p

    # currently only analyse right stim cc
    cc_sig_right = pd.DataFrame(cc_sig[:, :, 1].reshape((-1,)))
    print(name, cc_sig_right.describe())
#    ioutils.write_hdf5(results_dir / f"naive_{name}_cc_sig_right_stim_{duration}.h5", cc_sig_right)

    # left, positively above the cc-threshold
    left_pos_cc, left_pos_m2, left_pos_ppc = correlation.cc_matrix(
        m2_units, ppc_units, cc_sig, cc_threshold=0.25, s=0, naive=True, pos=True
    )

    left_pos_cc_max.append(correlation.find_max_cc(left_pos_cc, pos=True))
    left_pos_m2_counts.append(left_pos_m2["Count"])
    left_pos_ppc_counts.append(left_pos_ppc["Count"])

    # left, negatively above the cc-threshold
    left_neg_cc, left_neg_m2, left_neg_ppc = correlation.cc_matrix(
        m2_units, ppc_units, cc_sig, cc_threshold=0.25, s=0, naive=True, pos=False
    )

    left_neg_cc_max.append(correlation.find_max_cc(left_neg_cc, pos=False))
    left_neg_m2_counts.append(left_neg_m2["Count"])
    left_neg_ppc_counts.append(left_neg_ppc["Count"])

    # right, positively above the cc-threshold
    right_pos_cc, right_pos_m2, right_pos_ppc = correlation.cc_matrix(
        m2_units, ppc_units, cc_sig, cc_threshold=0.25, s=1, naive=True, pos=True
    )

    right_pos_cc_max.append(correlation.find_max_cc(right_pos_cc, pos=True))
    right_pos_m2_counts.append(right_pos_m2["Count"])
    right_pos_ppc_counts.append(right_pos_ppc["Count"])

    # right, negatively below the cc-threshold
    right_neg_cc, right_neg_m2, right_neg_ppc = correlation.cc_matrix(
        m2_units, ppc_units, cc_sig, cc_threshold=0.25, s=1, naive=True, pos=False
    )

    right_neg_cc_max.append(correlation.find_max_cc(right_neg_cc, pos=False))
    right_neg_m2_counts.append(right_neg_m2["Count"])
    right_neg_ppc_counts.append(right_neg_ppc["Count"])

assert False
# concat all sessions
left_pos_cc_max = correlation.df_max_cc(left_pos_cc_max)
left_neg_cc_max = correlation.df_max_cc(left_neg_cc_max)
right_pos_cc_max = correlation.df_max_cc(right_pos_cc_max)
right_neg_cc_max = correlation.df_max_cc(right_neg_cc_max)

max_cc = pd.concat(
    [left_pos_cc_max, left_neg_cc_max, right_pos_cc_max, right_neg_cc_max],
    keys=["left pos", "left neg", "right pos", "right neg"],
    names=["condition", "session"],
).T

'''
currently, directly comparable with trained mice are right stim trials. thus,
get a new set of m2&ppc sig corr unit counts that contains only right stim.
aligned.
'''
# get anatomical laterality from all left stim aligned trials
left_ipsi_pos_m2_counts = correlation.concat_unit_count(left_pos_m2_counts, ipsi=True)
left_ipsi_neg_m2_counts = correlation.concat_unit_count(left_neg_m2_counts, ipsi=True)
left_contra_pos_m2_counts = correlation.concat_unit_count(left_pos_m2_counts, ipsi=False)
left_contra_neg_m2_counts = correlation.concat_unit_count(left_neg_m2_counts, ipsi=False)

# get anatomical laterality from all right stim aligned trials
right_ipsi_pos_m2_counts = correlation.concat_unit_count(right_pos_m2_counts, ipsi=True)
right_ipsi_neg_m2_counts = correlation.concat_unit_count(right_neg_m2_counts, ipsi=True)
right_contra_pos_m2_counts = correlation.concat_unit_count(right_pos_m2_counts, ipsi=False)
right_contra_neg_m2_counts = correlation.concat_unit_count(right_neg_m2_counts, ipsi=False)

# do concatenation based on anatomical laterality, i.e., ipsi: left m2 & left ppc sessions
# currently only use right stim trials to make data comparable with expert sessions.
right_m2_counts = pd.DataFrame(
    [
        right_ipsi_pos_m2_counts,
        right_contra_pos_m2_counts,
        right_ipsi_neg_m2_counts,
        right_contra_neg_m2_counts,
    ],
    index=["ipsi pos", "contra pos", "ipsi neg", "contra neg"],
).T

# get anatomical laterality from all left stim aligned trials
left_ipsi_pos_ppc_counts = correlation.concat_unit_count(left_pos_ppc_counts, ipsi=True)
left_ipsi_neg_ppc_counts = correlation.concat_unit_count(left_neg_ppc_counts, ipsi=True)
left_contra_pos_ppc_counts = correlation.concat_unit_count(left_pos_ppc_counts, ipsi=False)
left_contra_neg_ppc_counts = correlation.concat_unit_count(left_neg_ppc_counts, ipsi=False)

# get anatomical laterality from all right stim aligned trials
right_ipsi_pos_ppc_counts = correlation.concat_unit_count(right_pos_ppc_counts, ipsi=True)
right_ipsi_neg_ppc_counts = correlation.concat_unit_count(right_neg_ppc_counts, ipsi=True)
right_contra_pos_ppc_counts = correlation.concat_unit_count(right_pos_ppc_counts, ipsi=False)
right_contra_neg_ppc_counts = correlation.concat_unit_count(right_neg_ppc_counts, ipsi=False)

right_ppc_counts = pd.DataFrame(
    [
        right_ipsi_pos_ppc_counts,
        right_contra_pos_ppc_counts,
        right_ipsi_neg_ppc_counts,
        right_contra_neg_ppc_counts,
    ],
    index=["ipsi pos", "contra pos", "ipsi neg", "contra neg"],
).T

if save_hdf5:
    if do_resps_corr:
        ioutils.write_hdf5(results_dir / f"naive_resps_max_cc_{duration}.h5", max_cc)
        ioutils.write_hdf5(results_dir / f'naive_resps_m2_sig_corr_units_count_{duration}.h5', m2_counts)
        ioutils.write_hdf5(results_dir / f'naive_resps_ppc_sig_corr_units_count_{duration}.h5', ppc_counts)
    else:
        ioutils.write_hdf5(results_dir / f"naive_max_cc_{duration}.h5", max_cc)
        ioutils.write_hdf5(results_dir / f"naive_m2_sig_corr_units_count_right_stim_{duration}.h5", right_m2_counts)
        ioutils.write_hdf5(results_dir / f"naive_ppc_sig_corr_units_count_right_stim_{duration}.h5", right_ppc_counts)

sns.boxplot(
    data=right_m2_counts,
)
sns.stripplot(
    data=right_m2_counts,
)
plt.ylabel("Counts")
plt.yticks(np.arange(0, 50, 5))
plt.title("Number of PPC units correlated with a M2 unit")
utils.save(fig_dir / f"m2_sig_corr_unit_counts_right_stim_naive.pdf")

plt.clf()
sns.boxplot(
    data=right_ppc_counts,
)
sns.stripplot(
    data=right_ppc_counts,
)
plt.ylabel("Counts")
plt.yticks(np.arange(0, 50, 5))
plt.title("Number of M2 units correlated with a PPC unit")
utils.save(fig_dir / f"ppc_sig_corr_unit_counts_right_stim_naive.pdf")

assert False
"""
plots:

- distribution of cc (histogram)

- cc heatmap of each session

- number of unit that a PPC/M2 unit is correlated to from M2/PPC, barplot & median boxplot

count the number of M2 units that are sig. correlated with PPC units, i.e.,
given a unit in M2, how many PPC units does it sig. correlated to, and same
in PPC-M2. Genenral overview of the functional correlation between PPC-M2
(not directional); categorise functional connection (correlation) into 1-1,
1-multi, and multi-1.  the returned m2 & ppc dataframe contains a list of
units in the correspondent area, and the number of unit from the other area
that it's sig. correlated to.
"""

#    _, axes = plt.subplots(2,1)
#    name = exp[session].name
#    # distribution of cc_sig
#    sns.histplot(
#        data=cc_sig[:, :, 0].reshape((-1,)),
#        ax=axes[0],
#    )
#    sns.histplot(
#        data=cc_sig[:, :, 1].reshape((-1,)),
#        ax=axes[1],
#    )
#    plt.ylim([0, 800])
##    plt.xlim([-0.8, 0.8])
#    plt.suptitle(name)
#    plt.gcf().set_size_inches(10, 20) #(width, height)
#    utils.save(fig_dir / f"correlation_coefficient_histo_naive_{name}.pdf", nosize=True)
#
#    name = exp[session].name
#    sns.heatmap(
#        data=sig_ipsi_pos_df,
#        vmin=0,
#        vmax=0.9,
#        ax=axes[0],
#    )
#    sns.heatmap(
#       data=sig_contra_pos_df,
#       vmin=0,
#       vmax=0.9,
#       ax=axes[1],
#   )
#
#    plt.suptitle(name)
#    plt.gcf().set_size_inches(10, 20)
#    utils.save(
#        fig_dir / f"pos_ipsi&contra_correlation_heatmap_naive_{name}.pdf", nosize=True
#    )

#    plt.clf()
#    _, axes = plt.subplots(4, len(exp), sharey=True)
#    name = exp[session].name
#    sns.barplot(
#        data=sig_ipsi_pos_m2,
#        x="M2 Unit ID",
#        y="Count",
#        ax=axes[0][session],
#    )
#    sns.barplot(
#        data=sig_contra_pos_m2,
#        x="M2 Unit ID",
#        y="Count",
#        ax=axes[1][session],
#    )
#    sns.barplot(
#        data=sig_ipsi_pos_ppc,
#        x="PPC Unit ID",
#        y="Count",
#        ax=axes[2][session],
#    )
#    sns.barplot(
#        data=sig_contra_pos_ppc,
#        x="PPC Unit ID",
#        y="Count",
#        ax=axes[3][session],
#    )
#    axes[0][session].set_title(name)
#    axes[0][session].set_xlabel(
#        "number of ppc units a m2 unit correlated to (ipsi stim)"
#    )
#    axes[1][session].set_xlabel(
#        "number of ppc units a m2 unit correlated to (contra stim)"
#    )
#    axes[2][session].set_xlabel(
#        "number of m2 units a ppc unit correlated to (ipsi stim)"
#    )
#    axes[3][session].set_xlabel(
#        "number of m2 units a ppc unit correlated to (contra stim)"
#    )
#    plt.gcf().set_size_inches(20, 15)
#    utils.save(
#        fig_dir / f"pos_ipsi&contra_unit_counts_bar_naive_{name}.pdf", nosize=True
#    )
#
#    _, axes = plt.subplots(4, len(exp), sharey=True)
#    name = exp[session].name
#    sns.boxplot(
#        data=sig_ipsi_pos_m2,
#        y="Count",
#        ax=axes[0][session],
#    )
#    sns.stripplot(
#        data=sig_ipsi_pos_m2,
#        y="Count",
#        ax=axes[0][session],
#    )
#    sns.boxplot(
#        data=sig_contra_pos_m2,
#        y="Count",
#        ax=axes[1][session],
#    )
#    sns.stripplot(
#        data=sig_contra_pos_m2,
#        y="Count",
#        ax=axes[1][session],
#    )
#    sns.boxplot(
#        data=sig_ipsi_pos_ppc,
#        y="Count",
#        ax=axes[2][session],
#    )
#    sns.stripplot(
#        data=sig_ipsi_pos_ppc,
#        y="Count",
#        ax=axes[2][session],
#    )
#    sns.boxplot(
#        data=sig_contra_pos_ppc,
#        y="Count",
#        ax=axes[3][session],
#    )
#    sns.stripplot(
#        data=sig_contra_pos_ppc,
#        y="Count",
#        ax=axes[3][session],
#    )
#
#    axes[0][session].set_title(name)
#    axes[0][0].set_xlabel(
#        "number of ppc units a m2 unit correlated to (ipsi stim)"
#    )
#    axes[1][0].set_xlabel(
#        "number of ppc units a m2 unit correlated to (contra stim)"
#    )
#    axes[2][0].set_xlabel(
#        "number of m2 units a ppc unit correlated to (ipsi stim)"
#    )
#    axes[3][0].set_xlabel(
#        "number of m2 units a ppc unit correlated to (contra stim)"
#    )
#    plt.yticks(np.arange(0, 50, 5))
#    plt.gcf().set_size_inches(20, 15)
#    utils.save(
#        fig_dir / f"pos_ipsi&contra_unit_counts_boxplot_naive_{name}.pdf", nosize=True
#    )
#
#    if (cc_sig[:, :, 0] <= -cc_threshold).any():
#        name = exp[session].name
#        plt.clf()
#
#        sns.heatmap(
#            data=sig_ipsi_neg_df,
#            vmin=-0.6, vmax=0,
#            cmap='YlGnBu',
#        )
#        plt.suptitle(name)
#        utils.save(fig_dir / f'ipsi_neg_correlation_heatmap_naive_{name}.pdf')
##
##        _, axes = plt.subplots(2,1, sharey=True)
##        name = exp[session].name
#        sns.barplot(
#            data=sig_ipsi_neg_m2,
#            x='M2 Unit ID',
#            y='Count',
#            ax=axes[0][0],
#            )
#        sns.barplot(
#            data=sig_ipsi_neg_ppc,
#            x='PPC Unit ID',
#            y='Count',
#            ax=axes[0][1],
#            )
#        sns.boxplot(
#            data=sig_ipsi_neg_m2,
#            y="Count",
#            ax=axes[0],
#        )
#        sns.stripplot(
#            data=sig_ipsi_neg_m2,
#            y="Count",
#            ax=axes[0],
#        )
#        sns.boxplot(
#            data=sig_ipsi_neg_ppc,
#            y="Count",
#            ax=axes[1],
#        )
#        sns.stripplot(
#            data=sig_ipsi_neg_ppc,
#            y="Count",
#            ax=axes[1],
#        )
#        axes[0].set_xlabel(
#            "number of ppc units a m2 unit correlated to (ipsi stim, neg cc)"
#        )
#        axes[1].set_xlabel(
#            "number of m2 units a ppc unit correlated to (ipsi stim, neg cc)"
#        )
#        plt.suptitle(name)
#        plt.yticks(np.arange(0, 18, 2))
#        plt.gcf().set_size_inches(10, 20)
#        utils.save(
#            fig_dir / f"ipsi_neg_correlation_unit_count_naive_{name}.pdf", nosize=True
#        )
#
#    if (cc_sig[:, :, 1] <= -cc_threshold).any():
#        plt.clf()
#        sns.heatmap(
#            data=sig_ipsi_neg_df,
#            vmin=-0.6, vmax=0,
#            cmap='YlGnBu',
#        )
#        plt.suptitle(name)
#        utils.save(fig_dir / f'contra_neg_correlation_heatmap_naive_{name}.pdf')

#        _, axes = plt.subplots(2,2, sharey=True)
#        name = exp[session].name
#        sns.barplot(
#            data=sig_contra_neg_m2,
#            x='M2 Unit ID',
#            y='Count',
#            ax=axes[0][0],
#            )
#        sns.barplot(
#            data=sig_contra_neg_ppc,
#            x='PPC Unit ID',
#            y='Count',
#            ax=axes[0][1],
#            )
#        _, axes = plt.subplots(2, 1, sharey=True)
#        sns.boxplot(
#            data=sig_contra_neg_m2,
#            y="Count",
#            ax=axes[0],
#        )
#        sns.stripplot(
#            data=sig_contra_neg_m2,
#            y="Count",
#            ax=axes[0],
#        )
#        sns.boxplot(
#            data=sig_contra_neg_ppc,
#            y="Count",
#            ax=axes[1],
#        )
#        sns.stripplot(
#            data=sig_contra_neg_ppc,
#            y="Count",
#            ax=axes[1],
#        )
#        axes[0].set_xlabel(
#            "number of ppc units a m2 unit correlated to (contra stim, neg cc)"
#        )
#        axes[1].set_xlabel(
#            "number of m2 units a ppc unit correlated to (contra stim, neg cc)"
#        )
#        plt.suptitle(name)
#        plt.yticks(np.arange(0, 18, 2))
#        plt.gcf().set_size_inches(10, 20)
#        utils.save(
#            fig_dir / f"contra_neg_correlation_unit_count_naive_{name}.pdf", nosize=True
#        )
##
##    cc_stats = {
##    "median": median,
##    "mean": mean,
##    "max": cc_max,
##    "min": cc_min,
#    "std": std,
#    "percentile": percentile,
#    }
#    # for a quick overview, use df.describe() to see all descriptive stats
#
#    cc_stats_df = pd.DataFrame(cc_stats).melt(value_name="numbers", var_name="stats")
#    print(name, cc_stats_df)

# ipsi_pos_m2_df = pd.concat(ipsi_pos_m2_list,ignore_index=True)
# ipsi_neg_m2_df = pd.concat(ipsi_neg_m2_list,ignore_index=True)
# contra_pos_m2_df = pd.concat(contra_pos_m2_list,ignore_index=True)
# contra_neg_m2_df = pd.concat(contra_neg_m2_list,ignore_index=True)
#
# ipsi_pos_ppc_df = pd.concat(ipsi_pos_ppc_list,ignore_index=True)
# ipsi_neg_ppc_df = pd.concat(ipsi_neg_ppc_list,ignore_index=True)
# contra_pos_ppc_df = pd.concat(contra_pos_ppc_list,ignore_index=True)
# contra_neg_ppc_df = pd.concat(contra_neg_ppc_list,ignore_index=True)
#
# sns.boxplot(
#    data=ipsi_pos_m2_df,
# )
# sns.stripplot(
#    data=ipsi_pos_m2_df,
# )
# plt.suptitle('number of ppc neurons that a m2 neurons is correlated to')
# plt.yticks(np.arange(0, 18, 2))
# plt.gcf().set_size_inches(5, 10)
# utils.save(
#    fig_dir / f"ipsi_pos_m2_unit_counts.pdf", nosize=True
# )
#
# sns.boxplot(
#    data=ipsi_neg_m2_df,
# )
# sns.stripplot(
#    data=ipsi_neg_m2_df,
# )
# plt.suptitle('number of ppc neurons that a m2 neurons is correlated to')
# plt.yticks(np.arange(0, 18, 2))
# plt.gcf().set_size_inches(5, 10)
# utils.save(
#    fig_dir / f"ipsi_neg_m2_unit_counts.pdf", nosize=True
# )
#
# sns.boxplot(
#    data=contra_pos_m2_df,
# )
# sns.stripplot(
#    data=contra_pos_m2_df,
# )
# plt.suptitle('number of ppc neurons that a m2 neurons is correlated to')
# plt.yticks(np.arange(0, 18, 2))
# plt.gcf().set_size_inches(5, 10)
# utils.save(
#    fig_dir / f"contra_pos_m2_unit_counts.pdf", nosize=True
# )
#
# sns.boxplot(
#    data=contra_neg_m2_df,
# )
# sns.stripplot(
#    data=contra_neg_m2_df,
# )
# plt.suptitle('number of ppc neurons that a m2 neurons is correlated to')
# plt.yticks(np.arange(0, 18, 2))
# plt.gcf().set_size_inches(5, 10)
# utils.save(
#    fig_dir / f"contra_neg_m2_unit_counts.pdf", nosize=True
# )
#
# sns.boxplot(
#    data=ipsi_pos_ppc_df,
# )
# sns.stripplot(
#    data=ipsi_pos_ppc_df,
# )
# plt.suptitle('number of m2 neurons that a ppc neurons is correlated to')
# plt.yticks(np.arange(0, 18, 2))
# plt.gcf().set_size_inches(5, 10)
# utils.save(
#    fig_dir / f"ipsi_pos_ppc_unit_counts.pdf", nosize=True
# )
#
# sns.boxplot(
#    data=ipsi_neg_ppc_df,
# )
# sns.stripplot(
#    data=ipsi_neg_ppc_df,
# )
# plt.suptitle('number of m2 neurons that a ppc neurons is correlated to')
# plt.yticks(np.arange(0, 18, 2))
# plt.gcf().set_size_inches(5, 10)
# utils.save(
#    fig_dir / f"ipsi_neg_ppc_unit_counts.pdf", nosize=True
# )
#
# sns.boxplot(
#    data=contra_pos_ppc_df,
# )
# sns.stripplot(
#    data=contra_pos_ppc_df,
# )
# plt.suptitle('number of m2 neurons that a ppc neurons is correlated to')
# plt.yticks(np.arange(0, 18, 2))
# plt.gcf().set_size_inches(5, 10)
# utils.save(
#    fig_dir / f"contra_pos_ppc_unit_counts.pdf", nosize=True
# )
#
# sns.boxplot(
#    data=contra_neg_ppc_df,
# )
# sns.stripplot(
#    data=contra_neg_ppc_df,
# )
# plt.suptitle('number of m2 neurons that a ppc neurons is correlated to')
# plt.yticks(np.arange(0, 18, 2))
# plt.gcf().set_size_inches(5, 10)
# utils.save(
#    fig_dir / f"contra_neg_ppc_unit_counts.pdf", nosize=True
# )
