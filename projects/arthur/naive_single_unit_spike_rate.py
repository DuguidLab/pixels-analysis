'''
Naive Mice
====

plot spike rate trace of the ppc-m2 units pair with the highest correlation coefficient.
'''

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pixels import Experiment, ioutils
from pixels.behaviours.reach import VisualOnly, ActionLabels, Events
from pixtools import spike_rate, utils, correlation

mice = [       
    'HFR20',
    'HFR22',
    'HFR23',
]

exp = Experiment(
    mice,
    VisualOnly,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

fig_dir = Path('~/duguidlab/visuomotor_control/AZ_notes/npx-plots/naive')
results_dir = Path("~/pixels-analysis/projects/arthur/results")

# Select units
duration = 2
units = exp.select_units(
        min_depth=0, max_depth=1200,
        name="cortex0-1200"
        )

# get spike rate for left & right visual stim.
print('get spike rates...')
#stim_left = exp.align_trials(
#    ActionLabels.naive_left,
#    Events.led_on,
#    'spike_rate',
#    units=units,
#    duration=duration,
#)
stim_right = exp.align_trials(
    ActionLabels.naive_right,
    Events.led_on,
    'spike_rate',
    units=units,
    duration=duration,
)

max_cc = ioutils.read_hdf5(results_dir / 'naive_max_cc_2.h5')

spike_rate.single_unit_spike_rate(
    data=stim_right[4][0][618],
    cell_id=618,
)
spike_rate.single_unit_spike_rate(
    data=stim_right[4][1][423],
    cell_id=423,
)
plt.suptitle('naive Max.right stim neg cc M2-PPC Pair')
plt.gcf().set_size_inches(5, 5)
utils.save(fig_dir / f'naive_max_right_neg_cc_pair_spike_rate.pdf', nosize=True)

session_idx_pos, m2_pos_max, ppc_pos_max = correlation.max_cc_ids(
    max_cc=max_cc,
    pos=True
)

assert False
_,axes = plt.subplots(2, 1, sharex=True)

# this pair has the highest pos cc
spike_rate.single_unit_spike_rate(
    data=stim_right[session_idx_pos][0][m2_pos_max],
    cell_id=m2_pos_max,
    ax=axes[0]
)
spike_rate.single_unit_spike_rate(
    data=stim_right[session_idx_pos][1][ppc_pos_max],
    cell_id=ppc_pos_max,
    ax=axes[0]
)
#utils.save(fig_dir / 'naive_highest_pos_cc_pair_spike_rate.pdf')

# this pair has the highest neg cc
session_idx_neg, m2_neg_max, ppc_neg_max = correlation.max_cc_ids(
    max_cc=max_cc,
    pos=False
)

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
utils.save(fig_dir / f'naive_max_pos&neg_cc_pair_spike_rate.pdf', nosize=True)
