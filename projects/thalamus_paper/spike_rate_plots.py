from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_rate


mice = [
    'C57_724',
    'C57_1288723',
    'C57_1288727',
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


## Spike rate plots
hits = exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    'spike_rate',
    duration=duration,
)

stim = exp.align_trials(
    ActionLabels.uncued_laser_push_full,
    Events.back_sensor_open,
    'spike_rate',
    duration=duration,
)

stim_miss = exp.align_trials(
    ActionLabels.uncued_laser_nopush,
    Events.laser_onset,
    'spike_rate',
    duration=duration,
)

for session in range(len(exp)):
    # per unit
    fig = spike_rate.per_unit_plot(hits[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to cued push)')
    utils.save(fig_dir / f'unit_spike_rate_cued_push_{duration}s_{name}')

    fig = spike_rate.per_unit_plot(stim[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to stim push)')
    utils.save(fig_dir / f'unit_spike_rate_stim_push_{duration}s_{name}')

    fig = spike_rate.per_unit_plot(stim_miss[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to nopush stim)')
    utils.save(fig_dir / f'unit_spike_rate_stim-miss_{duration}s_{name}')

    # per trial
    fig = spike_rate.per_trial_plot(hits[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to cued push)')
    utils.save(fig_dir / f'trial_spike_rate_cued_push_{duration}s_{name}')

    fig = spike_rate.per_trial_plot(stim[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to stim push)')
    utils.save(fig_dir / f'trial_spike_rate_stim_push_{duration}s_{name}')

    fig = spike_rate.per_trial_plot(stim_miss[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to nopush stim)')
    utils.save(fig_dir / f'trial_spike_rate_stim-miss_{duration}s_{name}')
