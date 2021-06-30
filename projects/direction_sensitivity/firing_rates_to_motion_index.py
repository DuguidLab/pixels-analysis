# Plot MI aligned to tone

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

for i, session in enumerate(exp):
    name = session.name

    # pushes
    spike_rate.per_unit_spike_rate(pushes[i][rec_num], ci='sd')
    plt.suptitle(f'Pushes - per-unit across-trials firing rate (aligned to MI)')
    utils.save(fig_dir / f'pushes_unit_spike_rate_{duration}s_{name}.pdf')

    # pulls
    spike_rate.per_unit_spike_rate(pulls[i][rec_num], ci='sd')
    plt.suptitle(f'Pulls - per-unit across-trials firing rate (aligned to MI)')
    utils.save(fig_dir / f'pulls_unit_spike_rate_{duration}s_{name}.pdf')
