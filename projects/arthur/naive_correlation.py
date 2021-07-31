"""
Pairwise unit correlation between PPC & M2.

stim_left & stim_right are defined in naive_ipsi_contra_spike_rate.py. 
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.stats as sms
import seaborn as sns
import math
from naive_ipsi_contra_spike_rate import *

from scipy.stats import pearsonr
from statsmodels.stats import multitest

# set subplots axes
sns.color_palette("icefire", as_cmap=True)
ipsi_pos_m2_list = []
ipsi_pos_ppc_list = []
ipsi_neg_m2_list = []
ipsi_neg_ppc_list = []
contra_pos_m2_list = []
contra_pos_ppc_list = []
contra_neg_m2_list = []
contra_neg_ppc_list = []

for session in range(len(exp)):
    print(exp[session].name)

    stims = [
        stim_left[session],
        stim_right[session],
    ]

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
                    cc, p = pearsonr(a_trials, y=b_trials)

                    # if most values are constant, Pearson r returns NaN. Replace NaN cc by 0, and NaN p-values by 1.
                    if math.isnan(cc):
                        print("nan CC:", m2_unit, ppc_unit)
                        cc = 0
                    if math.isnan(p):
                        print("nan p:", m2_unit, ppc_unit)
                        p = 1

                    results_cc[a, b, s] = cc
                    results_p[a, b, s] = p

        # correct p-values by FDR (false discovery rate), where FDR=FP/(FP+TP). returned results_p is boolean, True means alpha<0.05.
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
    ##  sns.histplot(
    #       cc_sig.reshape((-1,)),
    #       )
    #
    # threshold of correlation coefficient for further analysis
    cc_threshold = 0.25

    sig_ipsi_pos = []
    sig_contra_pos = []
    sig_ipsi_neg = []
    sig_contra_neg = []
    median = []
    mean = []
    cc_max = []
    cc_min = []
    std = []
    percentile = []
    median.append((np.median(cc_sig[:, :, 0]), np.median(cc_sig[:, :, 1])))
    mean.append((np.mean(cc_sig[:, :, 0]), np.mean(cc_sig[:, :, 1])))
    cc_max.append((np.max(cc_sig[:, :, 0]), np.max(cc_sig[:, :, 1])))
    cc_min.append((np.min(cc_sig[:, :, 0]), np.min(cc_sig[:, :, 1])))
    std.append((np.std(cc_sig[:, :, 0]), np.std(cc_sig[:, :, 1])))
    percentile.append(
        (
            np.percentile(cc_sig[:, :, 0], [2.5, 97.5]),
            np.percentile(cc_sig[:, :, 1], [2.5, 97.5]),
        )
    )

    # ipsi, positively above the cc-threshold
    sig_idx_ipsi_pos = np.where((cc_sig[:, :, 0] >= cc_threshold))
    cc_sig_ipsi_pos = cc_sig[:, :, 0][
        sig_idx_ipsi_pos
    ]  # returns ipsi cc values, masked by index

    for i in range(len(sig_idx_ipsi_pos[0])):
        sig_ipsi_pos.append(
            ([m2_units[a] for a in sig_idx_ipsi_pos[0]][i], [ppc_units[b] for b in sig_idx_ipsi_pos[1]][i], cc_sig_ipsi_pos[i])
        )

    sig_ipsi_pos_df = pd.DataFrame(
        sig_ipsi_pos, columns=["M2 Unit ID", "PPC Unit ID", "Correlation Coefficient"]
    ).pivot(index="M2 Unit ID", columns="PPC Unit ID", values="Correlation Coefficient")
    

    sig_ipsi_pos_m2 = pd.DataFrame(
        dict(zip(*np.unique(sig_idx_ipsi_pos[0], return_counts=True))).items(),
        columns=["M2 Unit ID", "Count"],
    )
    sig_ipsi_pos_ppc = pd.DataFrame(
        dict(zip(*np.unique(sig_idx_ipsi_pos[1], return_counts=True))).items(),
        columns=["PPC Unit ID", "Count"],
    )
    ipsi_pos_m2_list.append(sig_ipsi_pos_m2['Count'])
    ipsi_pos_ppc_list.append(sig_ipsi_pos_ppc['Count'])

    # ipsi, negatively above the cc-threshold
    sig_idx_ipsi_neg = np.where((cc_sig[:, :, 0] <= -cc_threshold))
    cc_sig_ipsi_neg = cc_sig[:, :, 0][
        sig_idx_ipsi_neg
    ]  # returns ipsi cc values, masked by index

    for i in range(len(sig_idx_ipsi_neg[0])):
        sig_ipsi_neg.append(
            ([m2_units[a] for a in sig_idx_ipsi_neg[0]][i], [ppc_units[b] for b in sig_idx_ipsi_neg[1]][i], cc_sig_ipsi_neg[i])
        )

    sig_ipsi_neg_df = pd.DataFrame(
        sig_ipsi_neg, columns=["M2 Unit ID", "PPC Unit ID", "Correlation Coefficient"]
    ).pivot(index="M2 Unit ID", columns="PPC Unit ID", values="Correlation Coefficient")

    sig_ipsi_neg_m2 = pd.DataFrame(
        dict(zip(*np.unique(sig_idx_ipsi_neg[0], return_counts=True))).items(),
        columns=["M2 Unit ID", "Count"],
    )

    sig_ipsi_neg_ppc = pd.DataFrame(
        dict(zip(*np.unique(sig_idx_ipsi_neg[1], return_counts=True))).items(),
        columns=["PPC Unit ID", "Count"],
    )
    ipsi_neg_m2_list.append(sig_ipsi_neg_m2['Count'])
    ipsi_neg_ppc_list.append(sig_ipsi_neg_ppc['Count'])


    # contra, positively above the cc-threshold
    sig_idx_contra_pos = np.where((cc_sig[:, :, 1] >= cc_threshold))
    cc_sig_contra_pos = cc_sig[:, :, 1][
        sig_idx_contra_pos
    ]  # returns contra cc values, masked by index

    for i in range(len(sig_idx_contra_pos[0])):
        sig_contra_pos.append(
            ([m2_units[a] for a in sig_idx_contra_pos[0]][i], [ppc_units[b] for b in sig_idx_contra_pos[1]][i], cc_sig_contra_pos[i])
        )

    sig_contra_pos_df = pd.DataFrame(
        sig_contra_pos, columns=["M2 Unit ID", "PPC Unit ID", "Correlation Coefficient"]
    ).pivot(index="M2 Unit ID", columns="PPC Unit ID", values="Correlation Coefficient")

    sig_contra_pos_m2 = pd.DataFrame(
        dict(zip(*np.unique(sig_idx_contra_pos[0], return_counts=True))).items(),
        columns=["M2 Unit ID", "Count"],
    )
    sig_contra_pos_ppc = pd.DataFrame(
        dict(zip(*np.unique(sig_idx_contra_pos[1], return_counts=True))).items(),
        columns=["PPC Unit ID", "Count"],
    )
    contra_pos_m2_list.append(sig_contra_pos_m2['Count'])
    contra_pos_ppc_list.append(sig_contra_pos_ppc['Count'])

    # contra, negatively below the cc-threshold
    sig_idx_contra_neg = np.where((cc_sig[:, :, 0] <= -cc_threshold))
    cc_sig_contra_neg = cc_sig[:, :, 0][
        sig_idx_contra_neg
    ]  # returns contra cc values, masked by index

    for i in range(len(sig_idx_contra_neg[0])):
        sig_contra_neg.append(
            ([m2_units[a] for a in sig_idx_contra_neg[0]][i], [ppc_units[b] for b in sig_idx_contra_neg[1]][i], cc_sig_contra_neg[i])
        )

    sig_contra_neg_df = pd.DataFrame(
        sig_contra_neg, columns=["M2 Unit ID", "PPC Unit ID", "Correlation Coefficient"]
    ).pivot(index="M2 Unit ID", columns="PPC Unit ID", values="Correlation Coefficient")

    sig_contra_neg_m2 = pd.DataFrame(
        dict(zip(*np.unique(sig_idx_contra_neg[0], return_counts=True))).items(),
        columns=["M2 Unit ID", "Count"],
    )
    sig_contra_neg_ppc = pd.DataFrame(
        dict(zip(*np.unique(sig_idx_contra_neg[1], return_counts=True))).items(),
        columns=["PPC Unit ID", "Count"],
    )
    contra_neg_m2_list.append(sig_contra_neg_m2['Count'])
    contra_neg_ppc_list.append(sig_contra_neg_ppc['Count'])

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

    count the number of M2 units that are sig. correlated with PPC units, i.e.,
    given a unit in M2, how many PPC units does it sig. correlated to, and same
    in PPC-M2. Genenral overview of the functional correlation between PPC-M2
    (not directional); categorise functional connection (correlation) into 1-1,
    1-multi, and multi-1.  the returned m2 & ppc dataframe contains a list of
    units in the correspondent area, and the number of unit from the other area
    that it's sig. correlated to.
    """

    # all neg df are empty, thus removed.
    _, axes = plt.subplots(2,1)
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
#    plt.ylim([0, 1000])
##    plt.xlim([-0.8, 0.8])
#    plt.suptitle(name)
#    plt.gcf().set_size_inches(10, 20) #(width, height)
#    utils.save(fig_dir / f"correlation_coefficient_histo_naive_{name}.pdf", nosize=True)
#
    name = exp[session].name
    sns.heatmap(
        data=sig_ipsi_pos_df,
        vmin=0,
        vmax=0.9,
        ax=axes[0],
    )
    sns.heatmap(
       data=sig_contra_pos_df,
       vmin=0,
       vmax=0.9,
       ax=axes[1],
   )

    plt.suptitle(name)
    plt.gcf().set_size_inches(10, 20)
    utils.save(
        fig_dir / f"pos_ipsi&contra_correlation_heatmap_naive_{name}.pdf", nosize=True
    )

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
#
#        _, axes = plt.subplots(2,1, sharey=True)
#        name = exp[session].name
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
#
##        _, axes = plt.subplots(2,2, sharey=True)
##        name = exp[session].name
##        sns.barplot(
##            data=sig_contra_neg_m2,
##            x='M2 Unit ID',
##            y='Count',
##            ax=axes[0][0],
##            )
##        sns.barplot(
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

#ipsi_pos_m2_df = pd.concat(ipsi_pos_m2_list,ignore_index=True) 
#ipsi_neg_m2_df = pd.concat(ipsi_neg_m2_list,ignore_index=True) 
#contra_pos_m2_df = pd.concat(contra_pos_m2_list,ignore_index=True) 
#contra_neg_m2_df = pd.concat(contra_neg_m2_list,ignore_index=True) 
#
#ipsi_pos_ppc_df = pd.concat(ipsi_pos_ppc_list,ignore_index=True) 
#ipsi_neg_ppc_df = pd.concat(ipsi_neg_ppc_list,ignore_index=True) 
#contra_pos_ppc_df = pd.concat(contra_pos_ppc_list,ignore_index=True) 
#contra_neg_ppc_df = pd.concat(contra_neg_ppc_list,ignore_index=True) 
#
#sns.boxplot(
#    data=ipsi_pos_m2_df,
#)
#sns.stripplot(
#    data=ipsi_pos_m2_df,
#)
#plt.suptitle('number of ppc neurons that a m2 neurons is correlated to')
#plt.yticks(np.arange(0, 18, 2))
#plt.gcf().set_size_inches(5, 10)
#utils.save(
#    fig_dir / f"ipsi_pos_m2_unit_counts.pdf", nosize=True
#)
#
#sns.boxplot(
#    data=ipsi_neg_m2_df,
#)
#sns.stripplot(
#    data=ipsi_neg_m2_df,
#)
#plt.suptitle('number of ppc neurons that a m2 neurons is correlated to')
#plt.yticks(np.arange(0, 18, 2))
#plt.gcf().set_size_inches(5, 10)
#utils.save(
#    fig_dir / f"ipsi_neg_m2_unit_counts.pdf", nosize=True
#)
#
#sns.boxplot(
#    data=contra_pos_m2_df,
#)
#sns.stripplot(
#    data=contra_pos_m2_df,
#)
#plt.suptitle('number of ppc neurons that a m2 neurons is correlated to')
#plt.yticks(np.arange(0, 18, 2))
#plt.gcf().set_size_inches(5, 10)
#utils.save(
#    fig_dir / f"contra_pos_m2_unit_counts.pdf", nosize=True
#)
#
#sns.boxplot(
#    data=contra_neg_m2_df,
#)
#sns.stripplot(
#    data=contra_neg_m2_df,
#)
#plt.suptitle('number of ppc neurons that a m2 neurons is correlated to')
#plt.yticks(np.arange(0, 18, 2))
#plt.gcf().set_size_inches(5, 10)
#utils.save(
#    fig_dir / f"contra_neg_m2_unit_counts.pdf", nosize=True
#)
#
#sns.boxplot(
#    data=ipsi_pos_ppc_df,
#)
#sns.stripplot(
#    data=ipsi_pos_ppc_df,
#)
#plt.suptitle('number of m2 neurons that a ppc neurons is correlated to')
#plt.yticks(np.arange(0, 18, 2))
#plt.gcf().set_size_inches(5, 10)
#utils.save(
#    fig_dir / f"ipsi_pos_ppc_unit_counts.pdf", nosize=True
#)
#
#sns.boxplot(
#    data=ipsi_neg_ppc_df,
#)
#sns.stripplot(
#    data=ipsi_neg_ppc_df,
#)
#plt.suptitle('number of m2 neurons that a ppc neurons is correlated to')
#plt.yticks(np.arange(0, 18, 2))
#plt.gcf().set_size_inches(5, 10)
#utils.save(
#    fig_dir / f"ipsi_neg_ppc_unit_counts.pdf", nosize=True
#)
#
#sns.boxplot(
#    data=contra_pos_ppc_df,
#)
#sns.stripplot(
#    data=contra_pos_ppc_df,
#)
#plt.suptitle('number of m2 neurons that a ppc neurons is correlated to')
#plt.yticks(np.arange(0, 18, 2))
#plt.gcf().set_size_inches(5, 10)
#utils.save(
#    fig_dir / f"contra_pos_ppc_unit_counts.pdf", nosize=True
#)
#
#sns.boxplot(
#    data=contra_neg_ppc_df,
#)
#sns.stripplot(
#    data=contra_neg_ppc_df,
#)
#plt.suptitle('number of m2 neurons that a ppc neurons is correlated to')
#plt.yticks(np.arange(0, 18, 2))
#plt.gcf().set_size_inches(5, 10)
#utils.save(
#    fig_dir / f"contra_neg_ppc_unit_counts.pdf", nosize=True
#)
