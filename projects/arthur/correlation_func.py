import matplotlib.pyplot as plt
import statsmodels.stats as sms
import seaborn as sns
import math
from naive_ipsi_contra_spike_rate import *

from scipy.stats import pearsonr
from statsmodels.stats import multitest

# set subplots axes
sns.color_palette("icefire", as_cmap=True)

def naive_cc_matrix(m2_units, ppc_units, cc_sig, cc_threshold=0.25, s=0, pos=True):
    """
    get correlation coefficient matrix, units that are significantly correlated
    with each other from cortical areas, number of correlated units count from
    all sessions.
    ===
    parameters:

    m2_units/ppc_units: list of units with their spike rates, indexed by
    units[session][rec_num].

    cc_sig: significant correlation coefficient matrix calculated from cortical
    areas.

    cc_threshold: threshold applied to correlation coefficient, only cc >
    threshold & < -threshold will be further analysed. Default is 0.25.

    s: side of the visual stimlation. 0 is left, 1 is right.

    pos: positivity of the significant correlation coefficient. True means
    further analysis is on positive correlations, False is on negtative
    correlations. Default is True.
    ===
    return:
    
    cc_df: df contains unit ids from both areas, and their correlation
    coefficient.

    m2_df: df contains m2 unit ids and the number of ppc units that they are
    correlated with.

    ppc_df: df contains ppc unit ids and the number of m2 units that they are
    correlated with.

    all_m2_counts: concatenated df with the number of ppc units that m2 units
    are correlated with, from all sessions.

    all_ppc_counts: concatenated df with the number of m2 units that ppc units
    are correlated with, from all sessions.
    """

    for session in range(len(exp)):
        if pos == True:
            idx = np.where((cc_sig[:, :, s] >= cc_threshold))
        else:
            idx = np.where((cc_sig[:, :, s] <= -cc_threshold))

        cc = cc_sig[:, :, s][idx]
        m2_ids = [m2_units[a] for a in idx[0]]
        ppc_ids = [ppc_units[b] for b in idx[1]]

        all_m2_counts = []
        all_ppc_counts = []
        cc_matrix = []
        for i in range(len(idx[0])):
            cc_matrix.append((m2_ids[i], ppc_ids[i], cc[i]))

        cc_df = pd.DataFrame(
            cc_matrix, columns=["M2 Unit ID", "PPC Unit ID", "Correlation Coefficient"]
        ).pivot(index="M2 Unit ID", columns="PPC Unit ID", values="Correlation Coefficient")
        
        m2_df = pd.DataFrame(np.unique(m2_ids, return_counts=True), index=["M2 Unit ID", "Count"]).T
        ppc_df = pd.DataFrame(np.unique(ppc_ids, return_counts=True), index=["PPC Unit ID", "Count"]).T
        all_m2_counts.append(m2_df['Count'])
        all_ppc_counts.append(ppc_df['Count'])


    return cc_df, m2_df, ppc_df, all_m2_counts, all_ppc_counts


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

    left_pos_cc, left_pos_m2, left_pos_ppc, all_m2_counts, all_ppc_counts = naive_cc_matrix(m2_units, ppc_units, cc_sig, cc_threshold = 0.25, s=0, pos=True)
    assert False
    #all_m2_counts_df = pd.concat(all_m2_counts, ignore_index=True)
    #all_ppc_counts_df = pd.concat(all_ppc_counts, ignore_index=True)
