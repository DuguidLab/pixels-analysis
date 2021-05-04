from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.reach import VisualOnly, ActionLabels, Events
from pixtools import spike_rate, utils


mice = [       
    'HFR19',
    'HFR20',
    #'HFR21',  # poor quality session
    'HFR22',
    'HFR23',
]

exp = Experiment(
    mice,
    VisualOnly,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/AZ_notes')

# Envelope for plots
ci = "sd"

## Select units

rec_num = 0
duration = 2

select = {
    "min_depth": 200,
    "max_depth": 1200,
    #"min_spike_width": 0.4,
    "duration": duration,
}

area = ["M2", "PPC"][rec_num]

## Spike rate plots for all visual stimulations

#hits = exp.align_trials(
    #ActionLabels.correct_left | correct_right,
    #Events.led_on,
    #'spike_rate',
    #duration=duration,
#)

stim_all = exp.align_trials(
    ActionLabels.naive_left | ActionLabels.naive_right,
    Events.led_on,
    'spike_rate',
    **select,
)

## Spike rate plots for short & long visual stimulation separately

#stim_short = exp.align_trials(
#    ActionLabels.naive_short,
#    Events.led_on,
#    'spike_rate',
#    **select,
#)
#
#
#stim_long = exp.align_trials(
#    ActionLabels.naive_long,
#    Events.led_on,
#    'spike_rate',
#    **select,
#)
#
### Spike rate plots for left & right visual stimulation separately
#
#stim_left = exp.align_trials(
#    ActionLabels.naive_left,
#    Events.led_on,
#    'spike_rate',
#    **select,
#)
#
#stim_right = exp.align_trials(
#    ActionLabels.naive_right,
#    Events.led_on,
#    'spike_rate',
#    **select,
#)
#
### Spike rate plots for left & right, short & long visual stimulation separately
#
#stim_left_short = exp.align_trials(
#    ActionLabels.naive_left_short,
#    Events.led_on,
#    'spike_rate',
#    **select,
#)
#
#stim_left_long = exp.align_trials(
#    ActionLabels.naive_left_long,
#    Events.led_on,
#    'spike_rate',
#    **select,
#)
#
#stim_right_short = exp.align_trials(
#    ActionLabels.naive_right_short,
#    Events.led_on,
#    'spike_rate',
#    **select,
#)
#
#stim_right_long = exp.align_trials(
#    ActionLabels.naive_right_long,
#    Events.led_on,
#    'spike_rate',
#    **select,
#)

for session in range(len(exp)):
    # per unit
#   fig = spike_rate.per_unit_spike_rate(hits[session][rec_num])
#   name = exp[session].name
#   plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to cued reach)')
#   utils.save(fig_dir / f'unit_spike_rate_cued_reach_{duration}s_{area}_{name}')

    fig = spike_rate.per_unit_spike_rate(stim_all[session][rec_num], ci=ci)
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to all visual stimulations)')
    utils.save(fig_dir / f'unit_spike_rate_all_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_unit_spike_rate(stim_short[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to short visual stimulations)')
    #utils.save(fig_dir / f'unit_spike_rate_short_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_unit_spike_rate(stim_long[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to long visual stimulations)')
    #utils.save(fig_dir / f'unit_spike_rate_long_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_unit_spike_rate(stim_left[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to visual stimulations on the left)')
    #utils.save(fig_dir / f'unit_spike_rate_left_viusal_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_unit_spike_rate(stim_right[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to visual stimulation on the right)')
    #utils.save(fig_dir / f'unit_spike_rate_right_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_unit_spike_rate(stim_left_short[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to short visual stimulations on the left)')
    #utils.save(fig_dir / f'unit_spike_rate_left_short_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_unit_spike_rate(stim_left_long[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to long visual stimulation on the left)')
    #utils.save(fig_dir / f'unit_spike_rate_left_long_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_unit_spike_rate(stim_right_short[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to short visual stimulations on the right)')
    #utils.save(fig_dir / f'unit_spike_rate_right_short_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_unit_spike_rate(stim_right_long[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to long visual stimulation on the right)')
    #utils.save(fig_dir / f'unit_spike_rate_right_long_visual_stim_{duration}s_{area}_{name}')

    # per trial
    fig = spike_rate.per_trial_spike_rate(stim_all[session][rec_num], ci=ci)
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to all visual stimulations)')
    utils.save(fig_dir / f'trial_spike_rate_all_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_trial_spike_rate(stim_short[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to short visual stimulations)')
    #utils.save(fig_dir / f'trial_spike_rate_short_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_trial_spike_rate(stim_long[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to long visual stimulations)')
    #utils.save(fig_dir / f'trial_spike_rate_long_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_trial_spike_rate(stim_left[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to visual stimulations on the left)')
    #utils.save(fig_dir / f'trial_spike_rate_left_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_trial_spike_rate(stim_right[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to visual stimulations on the right)')
    #utils.save(fig_dir / f'trial_spike_rate_right_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_trial_spike_rate(stim_left_short[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to short visual stimulations on the left)')
    #utils.save(fig_dir / f'trial_spike_rate_left_short_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_trial_spike_rate(stim_left_long[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to long visual stimulations on the left)')
    #utils.save(fig_dir / f'trial_spike_rate_left_long_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_trial_spike_rate(stim_right_short[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to short visual stimulations on the right)')
    #utils.save(fig_dir / f'trial_spike_rate_right_short_visual_stim_{duration}s_{area}_{name}')

    #fig = spike_rate.per_trial_spike_rate(stim_right_long[session][rec_num], ci=ci)
    #name = exp[session].name
    #plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to long visual stimulations on the right)')
    #utils.save(fig_dir / f'trial_spike_rate_right_long_visual_stim_{duration}s_{area}_{name}')
