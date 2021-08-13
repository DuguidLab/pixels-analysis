from pathlib import Path

import numpy as np
import pandas as pd

from pixels import ioutils
from pixtools import utils, stats_test

results_dir = Path("~/pixels-analysis/projects/arthur/results")

naive_m2_resps_bias = ioutils.read_hdf5(
    results_dir / f"naive_m2_resps_units_bias_groups.h5"
)
naive_ppc_resps_bias = ioutils.read_hdf5(
    results_dir / f"naive_ppc_resps_units_bias_groups.h5"
)

# drop 'opposite' from further comparison
naive_m2_resps_bias = naive_m2_resps_bias[naive_m2_resps_bias.columns[0:3]]
naive_ppc_resps_bias = naive_ppc_resps_bias[naive_ppc_resps_bias.columns[0:3]]

print('\nM2 tests')
stats_test.ow_anova(naive_m2_resps_bias, num='proportion', cat='group')

print('\nPPC tests')
stats_test.ow_anova(naive_ppc_resps_bias, num='proportion', cat='group')

print('all done :)')
