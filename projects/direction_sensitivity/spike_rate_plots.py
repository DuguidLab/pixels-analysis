import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pixels.behaviours.pushpull import ActionLabels, Events
from pixtools import spike_rate, utils

from setup import exp, fig_dir, rec_num

sns.set(font_scale=0.4)
duration = 4

## Spike rate plots
pushes = exp.align_trials(
    ActionLabels.rewarded_push,
    Events.back_sensor_open,
    'spike_rate',
    duration=duration,
    uncurated=True,
)

pulls = exp.align_trials(
    ActionLabels.rewarded_pull,
    Events.front_sensor_open,
    'spike_rate',
    duration=duration,
    uncurated=True,
)

for session in range(len(exp)):
    # per unit
    fig = spike_rate.per_unit_spike_rate(pushes[session][rec_num], ci='sd')
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to cued push)')
    utils.save(fig_dir / f'unit_spike_rate_cued_push_{duration}s_{name}_sd')

    fig = spike_rate.per_unit_spike_rate(pulls[session][rec_num], ci='sd')
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to cued pull)')
    utils.save(fig_dir / f'unit_spike_rate_cued_pull_{duration}s_{name}_sd')

    # per trial
    fig = spike_rate.per_trial_spike_rate(pushes[session][rec_num], ci='sd')
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to cued push)')
    utils.save(fig_dir / f'trial_spike_rate_cued_push_{duration}s_{name}_sd')

    fig = spike_rate.per_trial_spike_rate(pulls[session][rec_num], ci='sd')
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to cued pull)')
    utils.save(fig_dir / f'trial_spike_rate_cued_pull_{duration}s_{name}_sd')
