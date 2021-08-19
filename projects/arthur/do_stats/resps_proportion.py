from pathlib import Path

import numpy as np
import pandas as pd

from pixels import ioutils
from pixtools import utils, stats_test

results_dir = Path("~/duguidlab/visuomotor_control/neuropixels/interim/results")

naive_m2_resps_counts = ioutils.read_hdf5(
    results_dir / f"naive_m2_resps_units_count.h5"
)
naive_ppc_resps_counts = ioutils.read_hdf5(
    results_dir / f"naive_ppc_resps_units_count.h5"
)
expert_m2_resps_counts = ioutils.read_hdf5(
    results_dir / f"expert_m2_resps_units_count.h5"
)
expert_HFR25_ppc_resps_counts = ioutils.read_hdf5(
    results_dir / f"expert_ppc_resps_units_count.h5"
)
expert_HFR29_ppc_resps_counts = ioutils.read_hdf5(
    results_dir / f"expert_right_ppc_resps_units_count.h5"
)
expert_ppc_resps_counts = pd.concat([expert_HFR25_ppc_resps_counts, expert_HFR29_ppc_resps_counts])

print('compare naive & expert m2 responsive proportions:')
m2_t, m2_p, m2_d = stats_test.get_t_p_d(naive_m2_resps_counts['responsive proportion'], expert_m2_resps_counts['Proportion of Responsive Neurons'], equal_var=True)

print('\ncompare naive & expert ppc responsive proportions:')
ppc_t, ppc_p, ppc_d = stats_test.get_t_p_d(naive_ppc_resps_counts['responsive proportion'], expert_ppc_resps_counts['Proportion of Responsive Neurons'], equal_var=False)

print('\ncompare naive m2-ppc responsive proportions:')
naive_t, naive_p, naive_d = stats_test.get_t_p_d(naive_m2_resps_counts['responsive proportion'], naive_ppc_resps_counts['responsive proportion'], equal_var=True)

print('\ncompare expert m2-ppc responsive proportions:')
expert_t, expert_p, expert_d = stats_test.get_t_p_d(expert_m2_resps_counts['Proportion of Responsive Neurons'], expert_ppc_resps_counts['Proportion of Responsive Neurons'], equal_var=False)

print('\nall done :)')
