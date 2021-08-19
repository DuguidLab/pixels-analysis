from pathlib import Path
import numpy as np
import pandas as pd

from pixels import ioutils
from pixtools import utils, stats_test

results_dir = Path("~/duguidlab/visuomotor_control/neuropixels/interim/results")

naive_right_m2_sig_corr_counts = ioutils.read_hdf5(
    results_dir / f"naive_m2_sig_corr_units_count_right_stim_2.h5"
)
naive_right_ppc_sig_corr_counts = ioutils.read_hdf5(
    results_dir / f"naive_ppc_sig_corr_units_count_right_stim_2.h5"
)

expert_m2_sig_corr_counts = ioutils.read_hdf5(
    results_dir / f"expert_m2_sig_corr_units_count_left_stim_2.h5"
)
expert_ppc_sig_corr_counts = ioutils.read_hdf5(
    results_dir / f"expert_ppc_sig_corr_units_count_left_stim_2.h5"
)
save_stats_summary = True

m2_sig_corr_counts = pd.concat(
    [naive_right_m2_sig_corr_counts, expert_m2_sig_corr_counts],
    axis=1,
    keys=[0, 1],
    names=["mouse group"], # naive:0, trained:1
)
ppc_sig_corr_counts = pd.concat(
    [naive_right_ppc_sig_corr_counts, expert_ppc_sig_corr_counts],
    axis=1,
    keys=[0, 1],
    names=["mouse group"],# naive:0, trained:1
)

# do m2 ipsi pos t test, REMEMBER TO DROP NAN!
print("\nM2: ipsilateral M2-PPC positive correlation under contralateral-to-m2 stim.")
m2_ipsi_pos_t, m2_ipsi_pos_p, m2_ipsi_pos_d = stats_test.get_t_p_d(
    m2_sig_corr_counts[0]["ipsi pos"].dropna(),
    m2_sig_corr_counts[1]["ipsi pos"].dropna(),
    equal_var=False,
)

# do m2 contra pos t test, REMEMBER TO DROP NAN!
print("\nM2: contralateral M2-PPC positive correlation under contralateral-to-m2 stim.")
m2_contra_pos_t, m2_contra_pos_p, m2_contra_pos_d = stats_test.get_t_p_d(
    m2_sig_corr_counts[0]["contra pos"].dropna(),
    m2_sig_corr_counts[1]["contra pos"].dropna(),
    equal_var=False,
)

# do m2 contra neg t test, REMEMBER TO DROP NAN!
print("\nM2: contralateral M2-PPC negative correlation under contralateral-to-m2 stim.")
m2_contra_neg_t, m2_contra_neg_p, m2_contra_neg_d = stats_test.get_t_p_d(
    m2_sig_corr_counts[0]["contra neg"].dropna(),
    m2_sig_corr_counts[1]["contra neg"].dropna(),
    equal_var=False, # naive violate normality assumption
)

# do ppc ipsi pos t test, REMEMBER TO DROP NAN!
print("\nPPC: ipsilateral M2-PPC positive correlation under contralateral-to-m2 stim.")
ppc_ipsi_pos_t, ppc_ipsi_pos_p, ppc_ipsi_pos_d = stats_test.get_t_p_d(
    ppc_sig_corr_counts[0]["ipsi pos"].dropna(),
    ppc_sig_corr_counts[1]["ipsi pos"].dropna(),
    equal_var=False, # expert violate normality assumption
)

# do ppc contra pos t test, REMEMBER TO DROP NAN!
print("\nPPC: contralateral M2-PPC positive correlation under contralateral stim.")
ppc_contra_pos_t, ppc_contra_pos_p, ppc_contra_pos_d = stats_test.get_t_p_d(
    ppc_sig_corr_counts[0]["contra pos"].dropna(),
    ppc_sig_corr_counts[1]["contra pos"].dropna(),
    equal_var=False, # expert violate normality assumption
)

# do ppc contra neg t test, REMEMBER TO DROP NAN!
print("\nPPC: contralateral M2-PPC negitive correlation under contralateral stim.")
ppc_contra_neg_t, ppc_contra_neg_p, ppc_contra_neg_d = stats_test.get_t_p_d(
    ppc_sig_corr_counts[0]["contra neg"].dropna(),
    ppc_sig_corr_counts[1]["contra neg"].dropna(),
    equal_var=False,
)
'''
discard contra neg t test, as in naive, contra neg only has 1 data point.
'''

if save_stats_summary:
    ioutils.write_hdf5(
        results_dir / f"m2_sig_corr_units_stats.h5", m2_sig_corr_counts.describe()
    )
    ioutils.write_hdf5(
        results_dir / f"ppc_sig_corr_units_stats.h5", ppc_sig_corr_counts.describe()
    )

print("\nall done :)")
