'''
Compare distribution of significant correlation coefficients between naive & expert mice.
'''

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pixels import ioutils
from pixtools import utils, stats_test

results_dir = Path("~/duguidlab/visuomotor_control/neuropixels/interim/results")

# get naive right stim cc_sigs
HFR20_0_right_stim_cc = ioutils.read_hdf5(
    results_dir / f"naive_210310_HFR20_cc_sig_right_stim_2.h5"
)
HFR20_1_right_stim_cc = ioutils.read_hdf5(
    results_dir / f"naive_210312_HFR20_cc_sig_right_stim_2.h5"
)
HFR22_0_right_stim_cc = ioutils.read_hdf5(
    results_dir / f"naive_210412_HFR22_cc_sig_right_stim_2.h5"
)
HFR22_1_right_stim_cc = ioutils.read_hdf5(
    results_dir / f"naive_210413_HFR22_cc_sig_right_stim_2.h5"
)
HFR23_0_right_stim_cc = ioutils.read_hdf5(
    results_dir / f"naive_210414_HFR23_cc_sig_right_stim_2.h5"
)
HFR23_1_right_stim_cc = ioutils.read_hdf5(
    results_dir / f"naive_210415_HFR23_cc_sig_right_stim_2.h5"
)

# concat ipsi & contra
naive_cc_sig_ipsi = pd.concat(
    [HFR20_0_right_stim_cc, HFR22_0_right_stim_cc, HFR23_0_right_stim_cc],
    ignore_index=True,
)
print('naive ipsi cc stats:\n', naive_cc_sig_ipsi.describe())
naive_cc_sig_contra = pd.concat(
    [HFR20_1_right_stim_cc, HFR22_1_right_stim_cc, HFR23_1_right_stim_cc],
    ignore_index=True,
)
print('\nnaive contra cc stats:\n', naive_cc_sig_contra.describe())

# get expert left stim cc_sig
expert_cc_sig_contra = ioutils.read_hdf5(
    results_dir / f"expert_210709_HFR25_cc_sig_left_stim_2.h5"
)
print('\nexpert contra cc stats:\n', expert_cc_sig_contra.describe())
expert_cc_sig_ipsi = ioutils.read_hdf5(
    results_dir / f"expert_210710_HFR25_cc_sig_left_stim_2.h5"
)
print('\nexpert ipsi cc stats:\n', expert_cc_sig_ipsi.describe())

print(
'''
Step 1: Compare means
'''
)
# compare ipsi & contra cc means with t test
print("\nIpsilateral cc naive-expert comparison under contralateral-to-m2 stim.")
ipsi_cc_t, ipsi_cc_p, ipsi_cc_d = stats_test.get_t_p_d(
    np.squeeze(naive_cc_sig_ipsi),
    np.squeeze(expert_cc_sig_ipsi),
    equal_var=True,
)
print("\nContralateral cc naive-expert comparison under contralateral-to-m2 stim.")
contra_cc_t, contra_cc_p, contra_cc_d = stats_test.get_t_p_d(
    np.squeeze(naive_cc_sig_contra),
    np.squeeze(expert_cc_sig_contra),
    equal_var=True,
)

print(
'''
Step 2: Compare variances
'''
)
F_ipsi, p_ipsi = stats_test.compare_var(
    np.squeeze(naive_cc_sig_ipsi),
    np.squeeze(expert_cc_sig_ipsi),
)
F_contra, p_contra = stats_test.compare_var(
    np.squeeze(naive_cc_sig_contra),
    np.squeeze(expert_cc_sig_contra),
)

print(
'''
Step 3: Compare skewness
'''
)
ipsi_skew_ci = stats_test.compare_skew(
    np.squeeze(naive_cc_sig_ipsi),
    np.squeeze(expert_cc_sig_ipsi),
    names=['naive ipsi', 'expert ipsi'],
)
contra_skew_ci = stats_test.compare_skew(
    np.squeeze(naive_cc_sig_contra),
    np.squeeze(expert_cc_sig_contra),
    names=['naive contra', 'expert contra'],
)

print(
'''
Step 4: Plot skewness
'''
)
skew_ci = pd.concat([ipsi_skew_ci, contra_skew_ci], axis=1)
#sns.stripplot(data=skew_ci, jitter=0)
sns.pointplot(data=skew_ci, capsize=0.05)
plt.ylim([0,1.2])
plt.show()
print('\nall done :)')
