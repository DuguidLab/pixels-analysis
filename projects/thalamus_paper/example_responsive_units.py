from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_rate, spike_times, utils

fig_dir = Path('~/duguidlab/visuomotor_control/figures')
out = fig_dir / 'example_responsive_units_raster+spike_rate'

plt.tight_layout()
sns.set(font_scale=0.4)
palette = sns.color_palette()

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
    sigma=50,
)
stim = exp.align_trials(
    ActionLabels.uncued_laser_push_full,
    Events.back_sensor_open,
    'spike_rate',
    duration=4,
    sigma=50,
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

sns.set_theme(style="ticks")
fig, axes = plt.subplots(4, 4)
axbase = 0
rec_num = 0

# convenience in case i want to just generate some units quickly
to_skip = [199, 110, 217]

for i in range(len(examples)):
    ses, unit = examples[i]
    ax = axes[axbase][i % 4]

    if unit in to_skip:
        ax.set_axis_off()
        ax = axes[axbase + 1][i % 4]
        ax.set_axis_off()
        continue

    num_stim = len(stim_times[ses][rec_num].columns.get_level_values('trial').unique())
    spike_times.single_unit_raster(hit_times[ses][rec_num][unit], ax=ax, sample=num_stim)
    spike_times.single_unit_raster(stim_times[ses][rec_num][unit], ax=ax, start=num_stim)
    ax.set_axis_off()
    ax.xaxis.set_label_text("")
    ax.yaxis.set_label_text("")

    ax = axes[axbase + 1][i % 4]
    spike_rate.single_unit_spike_rate(hits[ses][rec_num][unit], ax=ax)
    spike_rate.single_unit_spike_rate(stim[ses][rec_num][unit], ax=ax)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([-2, 0, 2])
    ax.xaxis.set_label_text("")
    ax.yaxis.set_label_text("")

    bottom, top = ax.get_ylim()
    d = top / 5
    bottom -= d
    top += d
    ax.set_ylim(bottom, top)

    if i == 3:
        axbase = 2

utils.save(out)
