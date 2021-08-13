'''
Expert Mice
====

plot spike rate trace of the ppc-m2 units pair with the highest correlation coefficient.
'''

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pixels import Experiment, ioutils
from pixels.behaviours.reach import Reach, ActionLabels, Events
from pixtools import spike_rate, utils, correlation

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
results_dir = Path("~/pixels-analysis/projects/arthur/results")

# Select units
duration = 2
units = exp.select_units(
        min_depth=0, max_depth=1200,
        name="cortex0-1200"
        )

# get spike rate for left visual stim.
stim_left = exp.align_trials(
    ActionLabels.miss_left,
    Events.led_on,
    'spike_rate',
    units=units,
    duration=duration,
)

max_cc = ioutils.read_hdf5(results_dir / 'expert_max_cc.h5')

session_idx_pos, m2_pos_max, ppc_pos_max = correlation.max_cc_ids(
    max_cc=max_cc,
    pos=True
)

_,axes = plt.subplots(2, 1, sharex=True)

# this pair has the highest pos cc
spike_rate.single_unit_spike_rate(
    data=stim_left[session_idx_pos][0][m2_pos_max],
    cell_id=m2_pos_max,
    ax=axes[0]
)
spike_rate.single_unit_spike_rate(
    data=stim_left[session_idx_pos][1][ppc_pos_max],
    cell_id=ppc_pos_max,
    ax=axes[0]
)

#utils.save(fig_dir / 'naive_highest_pos_cc_pair_spike_rate.pdf')

# this pair has the highest neg cc
session_idx_neg, m2_neg_max, ppc_neg_max = correlation.max_cc_ids(
    max_cc=max_cc,
    pos=False
)

assert False
#plt.clf()
spike_rate.single_unit_spike_rate(
    data=stim_left[session_idx_neg][0][m2_neg_max],
    cell_id=m2_neg_max,
    ax=axes[1]
)
spike_rate.single_unit_spike_rate(
    data=stim_left[session_idx_neg][1][ppc_neg_max],
    cell_id=ppc_neg_max,
    ax=axes[1]
)
axes[1].set_xlabel('Time to Visual Stim. (s)')

plt.suptitle('Max.pos & neg cc M2-PPC Pair')
plt.gcf().set_size_inches(5, 10)
utils.save(fig_dir / f'expert_max_pos&neg_cc_pair_spike_rate.pdf', nosize=True)


