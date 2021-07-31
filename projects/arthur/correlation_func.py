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

def naive_cc_matrix(m2_units, ppc_units, cc_sig, cc_threshold, side=0, pos=True):
    side = [0, 1]
    cc_matrix = []
    all_m2_counts = []
    all_ppc_counts = []

    for session in range(len(exp)):
        for s in side:
            if pos=True:
                idx = np.where((cc_sig[:, :, s] >= cc_threshold))
            else:
                idx = np.where((cc_sig[:, :, s] <= -cc_threshold))

            cc = cc_sig[:, :, s][idx]
            m2_ids = [m2_units[a] for a in idx[0]]
            ppc_ids = [ppc_units[b] for b in idx[1]]

        for i in range(len(idx)):
            cc_matrix.append((m2_ids[i], ppc_ids[i], cc[i]))

        cc_df = pd.DataFrame(
            cc_matrix, columns=["M2 Unit ID", "PPC Unit ID", "Correlation Coefficient"]
        ).pivot(index="M2 Unit ID", columns="PPC Unit ID", values="Correlation Coefficient")
        
        m2_df = pd.DataFrame(
            dict(zip(*np.unique(m2_ids, return_counts=True))).items(),
            columns=["M2 Unit ID", "Count"],
        )
        assert False
        ppc_df = pd.DataFrame(
            dict(zip(*np.unique(ppc_ids, return_counts=True))).items(),
            columns=["PPC Unit ID", "Count"],
        )
        all_m2_counts.append(m2_df['Count'])
        all_ppc_counts.append(ppc_df['Count'])

    all_m2_counts_df = pd.concat(all_m2_counts, ignore_index=True)
    all_ppc_counts_df = pd.concat(all_ppc_counts, ignore_index=True)

    return cc_df, m2_df, ppc_df, all_m2_counts, all_ppc_counts
