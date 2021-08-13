from pathlib import Path
import math
from scipy.stats import pearsonr
import statsmodels.stats as sms
from statsmodels.stats import multitest

import pandas as pd
import numpy as np

from pixels import Experiment, ioutils
from pixels.behaviours.reach import Reach, ActionLabels, Events
from pixtools import spike_rate, utils

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
results_dir = Path('~/pixels-analysis/projects/arthur/results')

# Select units
print('get units...')
duration = 2
units = exp.select_units(
        min_depth=0, max_depth=1200,
        name="cortex0-1200"
        )

# get spike rate for left & right visual stim.
print('get units spike rates...')
stim_left = exp.align_trials(
    ActionLabels.miss_left,
    Events.led_on,
    'spike_rate',
    units=units,
    duration=duration,
)

expert_m2_resps = ioutils.read_hdf5(
    results_dir / f"expert_m2_resps_units.h5"
)
expert_ppc_resps = ioutils.read_hdf5(
    results_dir / f"expert_ppc_resps_units.h5"
)

# get spike rates from resps m2 units so i can do correlation between resps units from areas
for session in range(len(exp)):
    print(exp[session].name)

    m2_units = expert_m2_resps[session].dropna()
    ppc_units = expert_ppc_resps[session].dropna()

    # cache p-value & cc matrix
    cache_file_p = exp[session].interim / "cache" / "expert_resps_correlation_results_p.npy"
    cache_file_cc = exp[session].interim / "cache" / "expert_resps_correlation_results_cc.npy"
    if cache_file_p.exists() and cache_file_cc.exists():
        results_p = np.load(cache_file_p)
        results_cc = np.load(cache_file_cc)

    else:
        results_cc = np.zeros((len(m2_units), len(ppc_units)))
        results_p = np.zeros((len(m2_units), len(ppc_units)))

        for a, m2_unit in enumerate(m2_units):
            a_trials = stim_left[session][0][m2_unit]
            # reduce dimension
            a_trials = np.squeeze(a_trials.values.reshape((-1, 1)))

            for b, ppc_unit in enumerate(ppc_units):
                b_trials = stim_left[session][1][ppc_unit]
                # reduce dimension
                b_trials = np.squeeze(b_trials.values.reshape((-1, 1)))

                cc, p = pearsonr(a_trials, y=b_trials)

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

assert False
