from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_times, utils


mice = [
    'C57_724',
    'C57_1288723',
    'C57_1288727',
    'C57_1313404',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures')
duration = 4
rec_num = 0

select = {
    "min_depth": 500,
    "max_depth": 1200,
    "min_spike_width": 0.4,
    "duration": duration,
}

hits = exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    'spike_times',
    **select,
)

stim = exp.align_trials(
    ActionLabels.uncued_laser_push_full,
    Events.back_sensor_open,
    'spike_times',
    **select,
)

plt.tight_layout()
palette = sns.color_palette()

for session in range(len(exp)):
    num_stim = len(stim[session][rec_num].columns.get_level_values('trial').unique())
    subplots = spike_times.per_unit_raster(
        hits[session][rec_num], sample=num_stim, label=False
    )
    spike_times.per_unit_raster(
        stim[session][rec_num], start=num_stim, subplots=subplots
    )

    subplots.to_label.set_xticks([-2000, 0, 2000])
    subplots.to_label.set_xticklabels([-2, 0, 2])
    subplots.legend.text(
        0, 0.6,
        'cued pushes',
        transform=subplots.legend.transAxes,
        color=palette[0],
    )
    subplots.legend.text(
        0, 0.3,
        'opto-stim pushes',
        transform=subplots.legend.transAxes,
        color=palette[1],
    )

    name = exp[session].name
    plt.suptitle(f'Session {name} - pyramidal - per-unit spike times (aligned to push)')
    utils.save(fig_dir / f'unit_raster_PC_cued+stim_push_{duration}s_{name}.png')
