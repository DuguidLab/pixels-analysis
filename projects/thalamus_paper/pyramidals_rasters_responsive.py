import os

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pixels import Experiment, signal
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_times


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

def save(name):
    plt.gcf().savefig(
        Path('~/duguidlab/visuomotor_control/figures').expanduser() / name,
        bbox_inches='tight', dpi=300
    )


## FIRING RATES

duration = 4
rec_num = 0

select = {
    "min_depth": 500,
    "max_depth": 1200,
    "min_spike_width": 0.4,
}

hits = exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    'spike_times',
    duration=duration,
    **select,
)

stim = exp.align_trials(
    ActionLabels.uncued_laser_push_full,
    Events.back_sensor_open,
    'spike_times',
    duration=duration,
    **select,
)


hit_cis = exp.get_aligned_spike_rate_CI(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    slice(-99, 100),
    #slice(-249, 250),
    bl_event=Events.tone_onset,
    bl_win=slice(-199, 0),
    #bl_win=slice(-499, 0),
    **select,
)

stim_cis = exp.get_aligned_spike_rate_CI(
    ActionLabels.uncued_laser_push_full,
    Events.back_sensor_open,
    #slice(-99, 100),
    slice(-249, 250),
    bl_event=Events.laser_onset,
    #bl_win=slice(-199, 0),
    bl_win=slice(-499, 0),
    **select,
)

plt.tight_layout()
palette = sns.color_palette()

for session in range(len(exp)):

    # identify significantly responsive units
    resps = []
    for unit in hit_cis[session][rec_num].columns.values:
        t = hit_cis[session][rec_num][unit]
        if 0 < t[2.5] or t[97.5] < 0:
            resps.append(unit)

    num_stim = len(stim[session][rec_num].columns.get_level_values('trial').unique())
    subplots = spike_times.per_unit_raster(
        hits[session][rec_num][resps], sample=num_stim, label=False
    )
    spike_times.per_unit_raster(
        stim[session][rec_num][resps], start=num_stim, subplots=subplots
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
    save(f'unit_raster_responsive_PC_cued+stim_push_{duration}s_{name}.png')
    print(name, resps)
