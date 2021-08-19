"""
Pairwise unit correlation between PPC & M2, stim_left only.
"""

import matplotlib.pyplot as plt
import statsmodels.stats as sms
import seaborn as sns
import math

from scipy.stats import spearmanr
from statsmodels.stats import multitest
from pixels import Experiment, ioutils
from pixtools import spike_rate, utils, correlation

from pathlib import Path

import pandas as pd
import numpy as np

from pixels.behaviours.reach import Reach, ActionLabels, Events

mice = [       
    'HFR25',
]

exp = Experiment(
    mice,
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

fig_dir = Path('~/duguidlab/visuomotor_control/AZ_notes/npx-plots/expert')
results_dir = Path("~/duguidlab/visuomotor_control/neuropixels/interim/results")

save_hdf5 = False
# do correlations on responsive units only?
do_resps_corr = False

## Select units
duration = 2
units = exp.select_units(
        min_depth=0, max_depth=1200,
        name="cortex0-1200"
        )

# get spike rate for left & right visual stim.
stim_left = exp.align_trials(
    ActionLabels.miss_left,
    Events.led_on,
    'spike_rate',
    units=units,
    duration=duration,
)

# set subplots axes
cmap = sns.color_palette("icefire", as_cmap=True)

# get responsive units
expert_m2_resps = ioutils.read_hdf5(
    results_dir / f"expert_m2_resps_units.h5"
)
expert_ppc_resps = ioutils.read_hdf5(
    results_dir / f"expert_ppc_resps_units.h5"
)

left_pos_cc_max = []
left_neg_cc_max = []

left_pos_m2_counts = []
left_neg_m2_counts = []

left_pos_ppc_counts = []
left_neg_ppc_counts = []

for session in range(len(exp)):
    name = exp[session].name
    print(name)
    if do_resps_corr:
        m2_units = expert_m2_resps[session].dropna()
        ppc_units = expert_ppc_resps[session].dropna()
        cache_file_p = exp[session].interim / "cache" / f"expert_resps_correlation_results_p_{duration}.npy"
        cache_file_cc = exp[session].interim / "cache" / f"expert_resps_correlation_results_cc_{duration}.npy"

    else:
        cache_file_p = exp[session].interim / "cache" / f"correlation_results_p_{duration}.npy"
        cache_file_cc = exp[session].interim / "cache" / f"correlation_results_cc_{duration}.npy"
        m2_units = units[session][0]
        ppc_units = units[session][1]

    # cache p-value & cc matrix
    if cache_file_p.exists() and cache_file_cc.exists():
        results_p = np.load(cache_file_p)
        results_cc = np.load(cache_file_cc)

    else:
        results_cc = np.zeros((len(m2_units), len(ppc_units)))
        results_p = np.zeros((len(m2_units), len(ppc_units)))

        """
        Correlate unit a from M2 with b from PPC, with all spike rate trace
        concatenated, i.e., 'reduce' the dimension (time). side0=left, side1=right.
        determine ipsi/contra manually. naive M2 recording are all on left.
        """
        for a, m2_unit in enumerate(m2_units):
            a_trials = stim_left[session][0][m2_unit]
            # reduce dimension
            a_trials = np.squeeze(a_trials.values.reshape((-1, 1)))

            for b, ppc_unit in enumerate(ppc_units):
                b_trials = stim_left[session][1][ppc_unit]
                b_trials = np.squeeze(b_trials.values.reshape((-1, 1)))
                cc, p = spearmanr(a_trials, b_trials)

                # if most values are constant, Pearson r returns NaN. Replace
                # NaN cc by 0, and NaN p-values by 1.
                if math.isnan(cc):
                    print("nan CC:", m2_unit, ppc_unit)
                    cc = 0
                if math.isnan(p):
                    print("nan p:", m2_unit, ppc_unit)
                    p = 1

                results_cc[a, b] = cc
                results_p[a, b] = p

        # correct p-values by FDR (false discovery rate), where FDR=FP/(FP+TP).
        # returned results_p is boolean, True means alpha<0.05.
        results_p_corrected, _ = sms.multitest.fdrcorrection(
            results_p.reshape((-1,)), alpha=0.05
        )
        results_p = results_p_corrected.reshape(results_p.shape)
        results_p = results_p.astype(int)

        np.save(cache_file_p, results_p)
        np.save(cache_file_cc, results_cc)

    """
    Correlate unit a from M2 with b from PPC, with all spike rate trace
    concatenated, i.e., 'reduce' the dimension (time). side0=left, side1=right.
    determine ipsi/contra manually. naive M2 recording are all on left.
    """

    # filter correlation coefficient matrix by its p-value matrix, thus only
    # those pairs that are sig. correlated are left.
    cc_sig = results_cc * results_p
    cc_sig_df = pd.DataFrame(cc_sig.reshape((-1,)))
    print(name, cc_sig_df.describe())
#    ioutils.write_hdf5(results_dir / f"expert_{name}_cc_sig_left_stim_{duration}.h5", cc_sig_df)

    # left, positively above the cc-threshold
    left_pos_cc, left_pos_m2, left_pos_ppc = correlation.cc_matrix(m2_units, ppc_units, cc_sig, cc_threshold = 0.25, naive=False, pos=True)

    left_pos_cc_max.append(correlation.find_max_cc(left_pos_cc, pos=True))
    left_pos_m2_counts.append(left_pos_m2['Count'])
    left_pos_ppc_counts.append(left_pos_ppc['Count'])

    # left, negatively above the cc-threshold
    left_neg_cc, left_neg_m2, left_neg_ppc = correlation.cc_matrix(m2_units, ppc_units, cc_sig, cc_threshold = 0.25, naive=False, pos=False)

    left_neg_cc_max.append(correlation.find_max_cc(left_neg_cc, pos=False))
    left_neg_m2_counts.append(left_neg_m2['Count'])
    left_neg_ppc_counts.append(left_neg_ppc['Count']) # only first session has sig neg correlation

    plt.clf()
    name = exp[session].name
    sns.histplot(
        data=cc_sig.reshape((-1,)),
    )
    plt.ylim([0, 800])
    plt.xlim([-0.8, 0.8])
    plt.suptitle(name)
    utils.save(fig_dir / f"correlation_coefficient_expert_{name}.pdf")


assert False
'''
sessions are melted so it's easier to plot.
to retain session info, do count_series = pd.concat(left_pos_m2_counts,
ignore_index=True, keys=[0,1], names=['session'])
'''
left_pos_cc_max = correlation.df_max_cc(left_pos_cc_max)
left_neg_cc_max = correlation.df_max_cc(left_neg_cc_max)

max_cc = pd.concat(
    [left_pos_cc_max, left_neg_cc_max],
    keys=["left pos", "left neg"],
    names=["condition", "session"],
).T

# get cc based on anatomical laterality
left_ipsi_pos_m2_counts = left_pos_m2_counts[1]
left_ipsi_neg_m2_counts = left_neg_m2_counts[1]
left_contra_pos_m2_counts = left_pos_m2_counts[0]
left_contra_neg_m2_counts = left_neg_m2_counts[0]

left_m2_counts = pd.DataFrame(
    [
        left_ipsi_pos_m2_counts,
        #left_ipsi_neg_ppc_counts, # empty
        left_contra_pos_m2_counts,
        left_contra_neg_m2_counts,
    ],
    index = ['ipsi pos', 'contra pos', 'contra neg']).T

# get cc based on anatomical laterality
left_ipsi_pos_ppc_counts = left_pos_ppc_counts[1]
left_ipsi_neg_ppc_counts = left_neg_ppc_counts[1]
left_contra_pos_ppc_counts = left_pos_ppc_counts[0]
left_contra_neg_ppc_counts = left_neg_ppc_counts[0]

left_ppc_counts = pd.DataFrame(
    [
        left_ipsi_pos_ppc_counts,
        #left_ipsi_neg_ppc_counts, # empty
        left_contra_pos_ppc_counts,
        left_contra_neg_ppc_counts,
    ],
    index = ['ipsi pos', 'contra pos', 'contra neg']).T

plt.clf()
sns.boxplot(
    data=left_m2_counts,
)
sns.stripplot(
    data=left_m2_counts,
)
plt.ylabel("Counts")
plt.yticks(np.arange(0, 50, 5))
plt.title("Number of PPC units correlated with a M2 unit")
utils.save(fig_dir / f"m2_sig_corr_unit_counts_left_stim_expert.pdf")

plt.clf()
sns.boxplot(
    data=left_ppc_counts,
)
sns.stripplot(
    data=left_ppc_counts,
)
plt.ylabel("Counts")
plt.yticks(np.arange(0, 50, 5))
plt.title("Number of M2 units correlated with a PPC unit")
utils.save(fig_dir / f"ppc_sig_corr_unit_counts_left_stim_expert.pdf")

if save_hdf5:
    if do_resps_corr:
        ioutils.write_hdf5(results_dir / f"expert_resps_max_cc_{duration}.h5", max_cc)
        ioutils.write_hdf5(results_dir / f'expert_resps_m2_sig_corr_units_count_{duration}.h5', m2_counts)
        ioutils.write_hdf5(results_dir / f'expert_resps_ppc_sig_corr_units_count_{duration}.h5', ppc_counts)
    else:
        ioutils.write_hdf5(results_dir / f"expert_max_cc_{duration}.h5", max_cc)
        ioutils.write_hdf5(results_dir / f'expert_m2_sig_corr_units_count_left_stim_{duration}.h5', left_m2_counts)
        ioutils.write_hdf5(results_dir / f'expert_ppc_sig_corr_units_count_left_stim_{duration}.h5', left_ppc_counts)

assert False
"""
Matt's finding all pairs of units with cc>=threshold:
cc_sig_bool = cc_sig >= cc_threshold
good_rows_l = []
good_rows_r = []
for row in range(cc_sig.shape[0]):
    if cc_sig_bool[row, :, 0].any():
        good_rows_l.append(cc_sig[row, :, 0][..., None])
    if cc_sig_bool[row, :, 1].any():
        good_rows_r.append(cc_sig[row, :, 1][..., None])
cc_sig_reduced_l = np.concatenate(good_rows_l, axis=1).T
cc_sig_reduced_r = np.concatenate(good_rows_r, axis=1).T
assert cc_sig_reduced_l.shape[1] == cc_sig.shape[1]

good_cols_l = []
good_cols_r = []
for col in range(cc_sig.shape[1]):
    if cc_sig_bool[:, col, 0].any():
        good_cols_l.append(cc_sig_reduced_l[:, col][..., None])
    if cc_sig_bool[:, col, 1].any():
        good_cols_r.append(cc_sig_reduced_r[:, col][..., None])

cc_sig_reduced_l = np.concatenate(good_cols_l, axis=1)
cc_sig_reduced_r = np.concatenate(good_cols_r, axis=1)
"""

"""
plots:

- distribution of cc (histogram)

- cc heatmap of each session

- number of unit that a PPC/M2 unit is correlated to from M2/PPC, barplot & median boxplot
"""
    #plt.clf()
#    name = exp[session].name

    # pos plots
    #sns.histplot(
    #    data=cc_sig.reshape((-1,)),
    #)
    #plt.ylim([0, 1000])
    #plt.xlim([-0.8, 0.8])
    #plt.suptitle(name)
    #utils.save(fig_dir / f"correlation_coefficient_expert_{name}.pdf")

    #sns.heatmap(
    #    data=sig_pos_df,
    #    vmin=0, vmax=0.9,
    #)
    #plt.suptitle(name)
    #utils.save(fig_dir / f'pos_correlation_heatmap_expert_{name}.pdf')
    #
    #_, axes = plt.subplots(2, 1, sharey=True, sharex=True)
    #name = exp[session].name
    #    sns.barplot(
    #        data=sig_pos_m2,
    #        x='M2 Unit ID',
    #        y='Count',
    #        ax=axes[0][0],
    #        )
    #    sns.barplot(
    #        data=sig_pos_ppc,
    #        x='PPC Unit ID',
    #        y='Count',
    #        ax=axes[0][1],
    #        )
    #sns.boxplot(
    #    data=sig_pos_m2,
    #    y="Count",
    #    ax=axes[0],
    #)
    #sns.stripplot(
    #    data=sig_pos_m2,
    #    y="Count",
    #    ax=axes[0],
    #)
    #sns.boxplot(
    #    data=sig_pos_ppc,
    #    y="Count",
    #    ax=axes[1],
    #)
    #sns.stripplot(
    #    data=sig_pos_ppc,
    #    y="Count",
    #    ax=axes[1],
    #)
    #axes[0].set_xlabel(
    #    "number of ppc units a m2 unit correlated to (left stim, pos cc)"
    #)
    #axes[1].set_xlabel(
    #    "number of m2 units a ppc unit correlated to (left stim, pos cc)"
    #)
    #plt.suptitle(name)
    #plt.yticks(np.arange(0, 50, 5))
    #plt.gcf().set_size_inches(10, 20)
    #utils.save(fig_dir / f"pos_correlation_unit_count_expert_{name}.pdf", nosize=True)
    #
    ## sig_neg_m2 is empty for HFR25 session1
    #    # neg plots
 #   _, axes = plt.subplots(2, 1, sharey=True, sharex=True)
#    if (cc_sig <= -cc_threshold).any():
#        plt.clf()
#        sns.heatmap(
#            data=sig_neg_df,
#            vmin=-0.6, vmax=0,
#            cmap='YlGnBu',
#        )
#        utils.save(fig_dir / f'neg_correlation_heatmap_expert_{name}.pdf')
        #
        #        _, axes = plt.subplots(2,2, sharey=True)
        #        name = exp[session].name
        #        sns.barplot(
        #            data=sig_neg_m2,
        #            x='M2 Unit ID',
        #            y='Count',
        #            ax=axes[0][0],
        #            )
        #        sns.barplot(
        #            data=sig_neg_ppc,
        #            x='PPC Unit ID',
        #            y='Count',
        #            ax=axes[0][1],
        #            )
#        sns.boxplot(
#            data=sig_neg_m2,
#            y="Count",
#            ax=axes[0],
#        )
#        sns.stripplot(
#            data=sig_neg_m2,
#            y="Count",
#            ax=axes[0],
#        )
#        sns.boxplot(
#            data=sig_neg_ppc,
#            y="Count",
#            ax=axes[1],
#        )
#        sns.stripplot(
#            data=sig_neg_ppc,
#            y="Count",
#            ax=axes[1],
#        )
#        axes[0].set_xlabel(
#            "number of ppc units a m2 unit correlated to (left stim, neg cc)"
#        )
#        axes[1].set_xlabel(
#            "number of m2 units a ppc unit correlated to (left stim, neg cc)"
#        )
#    plt.suptitle(name)
##    plt.yticks(np.arange(0, 50, 5))
#    plt.gcf().set_size_inches(10, 20)
#    utils.save(
#        fig_dir / f"neg_correlation_unit_count_expert_{name}.pdf", nosize=True
#    )

"""
count the number of M2 units that are sig. correlated with PPC units, i.e.,
given a unit in M2, how many PPC units does it sig. correlated to, and same
in PPC-M2. Genenral overview of the functional correlation between PPC-M2
(not directional); categorise functional connection (correlation) into 1-1,
1-multi, and multi-1.  the returned m2 & ppc dataframe contains a list of
units in the correspondent area, and the number of unit from the other area
that it's sig. correlated to.
"""
    #cc_stats = {'median': median, 'mean': mean, 'max': cc_max, 'min': cc_min, 'std': std, 'percentile': percentile}
    #cc_stats_df = pd.DataFrame(cc_stats).melt(value_name="numbers", var_name="stats")
    #print(cc_stats_df)
