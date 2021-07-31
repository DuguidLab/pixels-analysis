"""
Pairwise unit correlation between PPC & M2, stim_left only.

stim_left & stim_right are defined in expert_ipsi_contra_spike_rate.py. 
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.stats as sms
import seaborn as sns
import math
from expert_ipsi_contra_spike_rate import *

from scipy.stats import pearsonr
from statsmodels.stats import multitest

# set subplots axes
cmap = sns.color_palette("icefire", as_cmap=True)
median = []
mean = []
cc_max = []
cc_min = []
std = []
percentile = []

for session in range(len(exp)):
    print(exp[session].name)

    # get our units[session][rec_num]
    m2_units = units[session][0]
    ppc_units = units[session][1]

    # cache p-value & cc matrix
    cache_file_p = exp[session].interim / "cache" / "correlation_results_p.npy"
    cache_file_cc = exp[session].interim / "cache" / "correlation_results_cc.npy"
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
                cc, p = pearsonr(a_trials, y=b_trials)

                # if most values are constant, Pearson r returns NaN. Replace NaN cc by 0, and NaN p-values by 1.
                if math.isnan(cc):
                    print("nan CC:", m2_unit, ppc_unit)
                    cc = 0
                if math.isnan(p):
                    print("nan p:", m2_unit, ppc_unit)
                    p = 1

                results_cc[a, b] = cc
                results_p[a, b] = p

        # correct p-values by FDR (false discovery rate), where FDR=FP/(FP+TP). returned results_p is boolean, True means alpha<0.05.
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

    median.append(np.median(cc_sig))
    mean.append(np.mean(cc_sig))
    cc_max.append(np.max(cc_sig))
    cc_min.append(np.min(cc_sig))
    std.append(np.std(cc_sig))
    percentile.append((np.percentile(cc_sig, 2.5), np.percentile(cc_sig, 97.5)))

    # threshold of correlation coefficient for further analysis
    cc_threshold = 0.25

    sig_pos = []
    sig_neg = []
    # ipsi, positively above the cc-threshold
    sig_idx_pos = np.where((cc_sig >= cc_threshold))
    cc_sig_pos = cc_sig[sig_idx_pos]  # returns ipsi cc values, masked by index
    m2_ids = [m2_units[a] for a in sig_idx_pos[0]]
    ppc_ids = [ppc_units[a] for a in sig_idx_pos[1]]

    for i in range(len(sig_idx_pos[0])):
        sig_pos.append((m2_ids[i], ppc_ids[i], cc_sig_pos[i]))

    sig_pos_df = pd.DataFrame(
        sig_pos, columns=["M2 Unit ID", "PPC Unit ID", "Correlation Coefficient"]
    ).pivot(index="M2 Unit ID", columns="PPC Unit ID", values="Correlation Coefficient")

    sig_pos_m2 = pd.DataFrame(
        dict(zip(*np.unique(sig_idx_pos[0], return_counts=True))).items(),
        columns=["M2 Unit ID", "Count"],
    )
    sig_pos_ppc = pd.DataFrame(
        dict(zip(*np.unique(sig_idx_pos[1], return_counts=True))).items(),
        columns=["PPC Unit ID", "Count"],
    )

    # ipsi, negatively above the cc-threshold
    sig_idx_neg = np.where((cc_sig <= -cc_threshold))
    cc_sig_neg = cc_sig[sig_idx_neg]  # returns ipsi cc values, masked by index
    m2_ids_neg = [m2_units[a] for a in sig_idx_neg[0]]
    ppc_ids_neg = [ppc_units[a] for a in sig_idx_neg[1]]

    for i in range(len(sig_idx_neg[0])):
        sig_neg.append((m2_ids_neg[i], ppc_ids_neg[i], cc_sig_neg[i]))

    sig_neg_df = pd.DataFrame(
        sig_neg, columns=["M2 Unit ID", "PPC Unit ID", "Correlation Coefficient"]
    ).pivot(index="M2 Unit ID", columns="PPC Unit ID", values="Correlation Coefficient")

    sig_neg_m2 = pd.DataFrame(
        dict(zip(*np.unique(sig_idx_neg[0], return_counts=True))).items(),
        columns=["M2 Unit ID", "Count"],
    )

    sig_neg_ppc = pd.DataFrame(
        dict(zip(*np.unique(sig_idx_neg[1], return_counts=True))).items(),
        columns=["PPC Unit ID", "Count"],
    )

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
    name = exp[session].name

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
    if (cc_sig <= -cc_threshold).any():
        plt.clf()
        sns.heatmap(
            data=sig_neg_df,
            vmin=-0.6, vmax=0,
            cmap='YlGnBu',
        )
        utils.save(fig_dir / f'neg_correlation_heatmap_expert_{name}.pdf')
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
    cc_stats = {'median': median, 'mean': mean, 'max': cc_max, 'min': cc_min, 'std': std, 'percentile': percentile}
    cc_stats_df = pd.DataFrame(cc_stats).melt(value_name="numbers", var_name="stats")
    print(cc_stats_df)
