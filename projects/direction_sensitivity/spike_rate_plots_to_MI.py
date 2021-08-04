# Plot MI aligned to tone

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pixels import Experiment
from pixels.behaviours.pushpull import ActionLabels, Events, PushPull
from pixtools import spike_rate, utils

duration = 4
rec_num = 0
fig_dir = Path('~/duguidlab/Direction_Sensitivity/neuropixels_figures')

mice = [       
    #"C57_1350950",  # no ROIs drawn
    "C57_1350951",  # MI done
    "C57_1350952",  # MI done
    #"C57_1350953",  # MI done
    "C57_1350954",  # MI done
    #"C57_1350955",  # no ROIs drawn
]

exp = Experiment(
    mice,
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
)

units = exp.select_units(
    min_depth=550,
    max_depth=1200,
    name="550-1200",
)

pushes = exp.align_trials(
    ActionLabels.rewarded_push_good_mi,
    Events.motion_index_onset,
    'spike_rate',
    duration=duration,
    units=units,
)

pulls = exp.align_trials(
    ActionLabels.rewarded_pull_good_mi,
    Events.motion_index_onset,
    'spike_rate',
    duration=duration,
    units=units,
)

# Only one recording
rec_num = 0

# I wanted to really check that the same units list is seen in all variables
all_unit_ids = [u for s in units for r in s for u in r]
ps_units = pushes.columns.get_level_values('unit').unique().values.copy()
pl_units = pulls.columns.get_level_values('unit').unique().values.copy()
ps_units.sort()
pl_units.sort()
assert all(ps_units == np.unique(all_unit_ids))
assert all(pl_units == np.unique(all_unit_ids))

# Plot
for i, session in enumerate(exp):
    name = session.name

    subplots = spike_rate.per_unit_spike_rate(pushes[i][rec_num], ci='sd')
    spike_rate.per_unit_spike_rate(pulls[i][rec_num], ci='sd', subplots=subplots)
    plt.suptitle(f'Pushes + pulls - per-unit firing rate (aligned to MI onset)')
    utils.save(fig_dir / f'pushpulls_unit_spike_rate_{duration}s_{name}.pdf')
