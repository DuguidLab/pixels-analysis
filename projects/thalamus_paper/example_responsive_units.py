from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_rate, spike_times, utils

plt.tight_layout()
sns.set(font_scale=0.4)
palette = sns.color_palette()
fig_dir = Path('~/duguidlab/visuomotor_control/figures')

mice = [
    'C57_1288723',
    'C57_1288727',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

hit_times = exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    'spike_times',
    duration=4,
)

stim_times = exp.align_trials(
    ActionLabels.uncued_laser_push_full,
    Events.back_sensor_open,
    'spike_times',
    duration=4,
)

hits = exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    'spike_rate',
    duration=4,
)
stim = exp.align_trials(
    ActionLabels.uncued_laser_push_full,
    Events.back_sensor_open,
    'spike_rate',
    duration=4,
)

examples = [
    (0, 159),
    (0, 130),
    (1, 199),
    (1, 169),
    (1, 110),
    (0, 177),
    (1, 137),
    (0, 217),
]

fig, axes = plt.subplots(4, 4)
axbase = 0
rec_num = 0

for i in range(len(examples)):
    ses, unit = examples[i]

    ax = axes[axbase][i % 4]
    num_stim = len(stim_times[ses][rec_num].columns.get_level_values('trial').unique())
    spike_times.single_unit_raster(hit_times[ses][rec_num][unit], ax=ax, sample=num_stim)
    spike_times.single_unit_raster(stim_times[ses][rec_num][unit], ax=ax, start=num_stim)

    ax = axes[axbase + 1][i % 4]
    spike_rate.single_unit_spike_rate(hits[ses][rec_num][unit], ax=ax)
    spike_rate.single_unit_spike_rate(stim[ses][rec_num][unit], ax=ax)
    bottom, top = ax.get_ylim()
    d = top / 5
    bottom -= d
    top += d
    ax.set_ylim(bottom, top)

    if i == 3:
        axbase = 2

utils.save(fig_dir / 'example_responsive_units_raster+spike_rate')
