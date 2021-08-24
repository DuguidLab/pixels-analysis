# Plot MI aligned to tone

import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from pixels.behaviours.pushpull import ActionLabels, Events
from pixtools import spike_rate, spike_times, utils

from setup import fig_dir, exp, pyramidals, rec_num

ci = 95
duration = 4
fig_dir = fig_dir / "spike_rates+rasters_by_resp_group"

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

pushes_times = exp.align_trials(
    ActionLabels.rewarded_push_good_mi,
    Events.motion_index_onset,
    'spike_times',
    duration=duration,
    units=pyramidals,
)

pulls_times = exp.align_trials(
    ActionLabels.rewarded_pull_good_mi,
    Events.motion_index_onset,
    'spike_times',
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
            s = math.ceil(math.sqrt(len(units) * 2)) + 1
            cols = s
            rows = s
            fig, axes = plt.subplots(rows, cols)

            num_pushes = len(pushes[i][rec_num].columns.get_level_values('trial').unique().values)

            for u, unit in enumerate(units):
                c = u % cols
                r = (u // cols) * 2
                ax1 = axes[r][c]
                ax2 = axes[r + 1][c]

                spike_rate.single_unit_spike_rate(
                    pushes[i][rec_num][unit], ax=ax1, ci=ci, cell_id=unit
                )
                spike_rate.single_unit_spike_rate(
                    pulls[i][rec_num][unit], ax=ax1, ci=ci
                )
                ax1.set_ylabel('')

                spike_times.single_unit_raster(
                    pushes_times[i][rec_num][unit], ax=ax2, unit_id=unit
                )
                spike_times.single_unit_raster(
                    pulls_times[i][rec_num][unit], ax=ax2, start=num_pushes,
                )

            plt.suptitle(f'Firing rates and rasters of units in group {name} (aligned to MI onset)')
            utils.save(fig_dir / f'spike_rate+raster_{name}_{duration}s_{session.name}.pdf')
