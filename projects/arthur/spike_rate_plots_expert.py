from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.reach import Reach, ActionLabels, Events
from pixtools import spike_rate, utils


mice = [       
    'HFR29', #first session only on right PPC
]

exp = Experiment(
    mice,
    Reach,
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
units = exp.select_units(
        min_depth=200, max_depth=1400,
        )

#area = ["M2", "PPC"][rec_num] #for all other recordings
area = ["PPC"][rec_num] #for HFR29 session 0

## Spike rate plots for all hits
#hits = exp.align_trials(
    #ActionLabels.correct_left | correct_right,
    #Events.led_on,
    #'spike_rate',
    #duration=duration,
#)

## Spike rate plots for all miss trials
miss_all = exp.align_trials(
    ActionLabels.miss_left | ActionLabels.miss_right,
    Events.led_on,
    'spike_rate',
    units=units,
    duration=duration,
)

#plot firing rate per unit & per trial
for session in range(len(exp)):
    # per unit
#   fig = spike_rate.per_unit_spike_rate(hits[session][rec_num])
#   name = exp[session].name
#   plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to cued reach)')
#   utils.save(fig_dir / f'unit_spike_rate_cued_reach_{duration}s_{area}_{name}')

#   m2 = stim_all[session][0]
#   ppc = stim_all[session][1]

# for HFR29 session 0 only
    ppc = miss_all[session][0]
    unit = ppc.columns.get_level_values('unit').unique()

#    for unit in ppc:
#        firing_rate = unit.mean()
#        firing_rate = firing_rate.iloc[0: 1000]
#        baseline = firing_rate.iloc[-500: -100]
#        firing_rate = firing_rate - baseline
#        if firing_rate.max() < abs(firing_rate.min()):
#            np.where(firing_rate == firing_rate.min())
#        else:
#            np.where(firing_rate == firing_rate.max())


    # per unit        
    fig = spike_rate.per_unit_spike_rate(miss_all[session][rec_num], ci=ci)
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to all missed trials)')
    utils.save(fig_dir / f'unit_spike_rate_all_visual_stim_{duration}s_{area}_{name}')

    # per trial
    fig = spike_rate.per_trial_spike_rate(miss_all[session][rec_num], ci=ci)
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to all missed trials)')
    utils.save(fig_dir / f'trial_spike_rate_all_visual_stim_{duration}s_{area}_{name}')

