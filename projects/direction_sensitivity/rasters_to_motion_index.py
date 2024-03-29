# Plot rasters aligned to MI onset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pixels.behaviours.pushpull import ActionLabels, Events
from pixtools import spike_times, utils

from setup import fig_dir, exp, pyramidals

sns.set(font_scale=0.4)
duration = 4

pushes = exp.align_trials(
    ActionLabels.rewarded_push_good_mi,
    Events.motion_index_onset,
    'spike_times',
    duration=duration,
    units=pyramidals,
)

pulls = exp.align_trials(
    ActionLabels.rewarded_pull_good_mi,
    Events.motion_index_onset,
    'spike_times',
    duration=duration,
    units=pyramidals,
)

# Only one recording
rec_num = 0

# I wanted to really check that the same units list is seen in all variables
all_unit_ids = [u for s in pyramidals for r in s for u in r]
ps_units = pushes.columns.get_level_values('unit').unique().values.copy()
pl_units = pulls.columns.get_level_values('unit').unique().values.copy()
ps_units.sort()
pl_units.sort()
assert all(ps_units == np.unique(all_unit_ids))
assert all(pl_units == np.unique(all_unit_ids))

plt.tight_layout()
palette = sns.color_palette()

# Plot
for session in range(len(exp)):
    subplots = spike_times.per_unit_raster(pushes[session][rec_num], label=False)
    num_pushes = len(pushes[session][rec_num].columns.get_level_values('trial').unique())
    spike_times.per_unit_raster(
        pulls[session][rec_num], start=num_pushes, subplots=subplots
    )

    subplots.to_label.set_xticks([-2000, 0, 2000])
    subplots.to_label.set_xticklabels([-2, 0, 2])
    subplots.legend.text(
        0, 0.6,
        'Pushes',
        transform=subplots.legend.transAxes,
        color=palette[0],
    )
    subplots.legend.text(
        0, 0.3,
        'Pulls',
        transform=subplots.legend.transAxes,
        color=palette[1],
    )

    name = exp[session].name
    plt.suptitle(f'Session {name} - pyramidal - per-unit spike times')
    utils.save(fig_dir / f'unit_raster_PC_{duration}s_{name}.png')
