# Plot MI aligned to tone

import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from pathlib import Path

from pixels import Experiment
from pixels.behaviours.pushpull import ActionLabels, Events, PushPull
from pixtools import spike_rate, utils

ci = 95
duration = 4
rec_num = 0
fig_dir = Path('~/duguidlab/Direction_Sensitivity/neuropixels_figures')
fig_dir = fig_dir / "spike_rates_by_resp_group"

mice = [       
    "C57_1350950",
    "C57_1350951",
    "C57_1350952",
    "C57_1350954",
]

exp = Experiment(
    mice,
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
)

pyramidals = exp.select_units(
    min_depth=550,
    max_depth=900,
    min_spike_width=0.4,
    name="550-900-pyramidals",
)

pushes = exp.align_trials(
    ActionLabels.rewarded_push_good_mi,
    Events.motion_index_onset,
    'spike_rate',
    duration=duration,
    units=pyramidals,
)

pulls = exp.align_trials(
    ActionLabels.rewarded_pull_good_mi,
    Events.motion_index_onset,
    'spike_rate',
    duration=duration,
    units=pyramidals,
)

# I wanted to really check that the same units list is seen in all variables
all_unit_ids = [u for s in pyramidals for r in s for u in r]
ps_units = pushes.columns.get_level_values('unit').unique().values.copy()
pl_units = pulls.columns.get_level_values('unit').unique().values.copy()
ps_units.sort()
pl_units.sort()
assert all(ps_units == np.unique(all_unit_ids))
assert all(pl_units == np.unique(all_unit_ids))

# Plot
for i, session in enumerate(exp):
    cached_groups = session.interim / "cache" / "responsive_groups.pickle"
    with cached_groups.open('rb') as fd:
        groups = pickle.load(fd)

    for name, units in groups.items():
        if units:
            plt.clf()
            subplots = spike_rate.per_unit_spike_rate(pushes[i][rec_num][units], ci=ci)
            spike_rate.per_unit_spike_rate(pulls[i][rec_num][units], ci=ci, subplots=subplots)
            plt.suptitle(f'Firing rates of units in group {name} (aligned to MI onset)')
            utils.save(fig_dir / f'spike_rate_{name}_{duration}s_{session.name}.pdf')
