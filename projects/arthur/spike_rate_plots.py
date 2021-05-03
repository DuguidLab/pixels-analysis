from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2

from pixels import Experiment
from pixels.behaviours.reach import Reach, ActionLabels, Events
from pixtools import spike_rate


mice = [       
 ##  'HFR19',
    'HFR20',
 ##  'HFR21',
 ##  'HFR22',
 ##  'HFR23',
]

exp = Experiment(
    mice,
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/AZ_notes')
duration = 4
rec_num = 1 


## Spike rate plots for all visual stimulations

#hits = exp.align_trials(
    #ActionLabels.correct_left | correct_right,
    #Events.led_on,
    #'spike_rate',
    #duration=duration,
#)

stim_all = exp.align_trials(
    ActionLabels.naive_left_short | ActionLabels.naive_left_long | ActionLabels.naive_right_short |ActionLabels.naive_right_long,
    Events.led_on,
    'spike_rate',
    duration=duration,
)

## Spike rate plots for short & long visual stimulation separately

stim_short = exp.align_trials(
    ActionLabels.naive_left_short | ActionLabels.naive_right_short,
    Events.led_on,
    'spike_rate',
    duration=duration,
)


stim_long = exp.align_trials(
    ActionLabels.naive_left_long | ActionLabels.naive_right_long,
    Events.led_on,
    'spike_rate',
    duration=duration,
)

## Spike rate plots for left & right visual stimulation separately

stim_left = exp.align_trials(
    ActionLabels.naive_left_short | ActionLabels.naive_left_long,
    Events.led_on,
    'spike_rate',
    duration=duration,
)

stim_right = exp.align_trials(
    ActionLabels.naive_right_short | ActionLabels.naive_right_long,
    Events.led_on,
    'spike_rate',
    duration=duration,
)

## Spike rate plots for left & right, short & long visual stimulation separately

stim_left_short = exp.align_trials(
    ActionLabels.naive_left_short,
    Events.led_on,
    'spike_rate',
    duration=duration,
)

stim_left_long = exp.align_trials(
    ActionLabels.naive_left_long,
    Events.led_on,
    'spike_rate',
    duration=duration,
)

stim_right_short = exp.align_trials(
    ActionLabels.naive_right_short,
    Events.led_on,
    'spike_rate',
    duration=duration,
)

stim_right_long = exp.align_trials(
    ActionLabels.naive_right_long,
    Events.led_on,
    'spike_rate',
    duration=duration,
)

#stim_miss = exp.align_trials(
#    ActionLabels.uncued_laser_nopush,
#    Events.laser_onset,
#    'spike_rate',
#    duration=duration,
#)

for session in range(len(exp)):
    # per unit
#   fig = spike_rate.per_unit_plot(hits[session][rec_num])
#   name = exp[session].name
#   plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to cued reach)')
#   utils.save(fig_dir / f'unit_spike_rate_cued_reach_{duration}s_{name}')

    fig = spike_rate.per_unit_plot(stim_all[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to all visual stimulations)')
    utils.save(fig_dir / f'unit_spike_rate_all_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_unit_plot(stim_short[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to short visual stimulations)')
    utils.save(fig_dir / f'unit_spike_rate_short_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_unit_plot(stim_long[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to long visual stimulations)')
    utils.save(fig_dir / f'unit_spike_rate_long_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_unit_plot(stim_left[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to visual stimulations on the left)')
    utils.save(fig_dir / f'unit_spike_rate_left_viusal_stim_{duration}s_{name}')

    fig = spike_rate.per_unit_plot(stim_right[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to visual stimulation on the right)')
    utils.save(fig_dir / f'unit_spike_rate_right_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_unit_plot(stim_left_short[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to short visual stimulations on the left)')
    utils.save(fig_dir / f'unit_spike_rate_left_short_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_unit_plot(stim_left_long[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to long visual stimulation on the left)')
    utils.save(fig_dir / f'unit_spike_rate_left_long_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_unit_plot(stim_right_short[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to short visual stimulations on the right)')
    utils.save(fig_dir / f'unit_spike_rate_right_short_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_unit_plot(stim_right_long[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to long visual stimulation on the right)')
    utils.save(fig_dir / f'unit_spike_rate_right_long_visual_stim_{duration}s_{name}')

    # per trial
    fig = spike_rate.per_trial_plot(stim_all[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to all visual stimulations)')
    utils.save(fig_dir / f'trial_spike_rate_all_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_trial_plot(stim_short[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to short visual stimulations)')
    utils.save(fig_dir / f'trial_spike_rate_short_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_trial_plot(stim_long[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to long visual stimulations)')
    utils.save(fig_dir / f'trial_spike_rate_long_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_trial_plot(stim_left[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to visual stimulations on the left)')
    utils.save(fig_dir / f'trial_spike_rate_left_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_trial_plot(stim_right[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to visual stimulations on the right)')
    utils.save(fig_dir / f'trial_spike_rate_right_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_trial_plot(stim_left_short[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to short visual stimulations on the left)')
    utils.save(fig_dir / f'trial_spike_rate_left_short_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_trial_plot(stim_left_long[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to long visual stimulations on the left)')
    utils.save(fig_dir / f'trial_spike_rate_left_long_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_trial_plot(stim_right_short[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to short visual stimulations on the right)')
    utils.save(fig_dir / f'trial_spike_rate_right_short_visual_stim_{duration}s_{name}')

    fig = spike_rate.per_trial_plot(stim_right_long[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to long visual stimulations on the right)')
    utils.save(fig_dir / f'trial_spike_rate_right_long_visual_stim_{duration}s_{name}')
