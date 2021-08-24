# Plot MI aligned to tone

import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from pixels.behaviours.pushpull import ActionLabels, Events
from pixtools import spike_rate, utils

from setup import fig_dir, exp, rec_num, pyramidals

ci = 95
duration = 4
fig_dir = fig_dir / "spike_rates_by_resp_group"

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
