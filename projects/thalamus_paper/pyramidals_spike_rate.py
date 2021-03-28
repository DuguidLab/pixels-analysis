from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import clusters, spike_rate, utils


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

## FIRING RATES

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
    'spike_rate',
    **select,
)

stim = exp.align_trials(
    ActionLabels.uncued_laser_push_full,
    Events.back_sensor_open,
    'spike_rate',
    **select,
)

for session in range(len(exp)):
    # per unit
    subplots = spike_rate.per_unit_spike_rate(hits[session][rec_num], ci='sd')
    spike_rate.per_unit_spike_rate(stim[session][rec_num], ci='sd', subplots=subplots)
    name = exp[session].name
    plt.suptitle(f'Session {name} - pyramidal - per-unit across-trials firing rate (aligned to push)')
    utils.save(fig_dir / f'unit_spike_rate_PC_cued+stim_push_{duration}s_{name}.png')

    # per trial
    subplots = spike_rate.per_trial_spike_rate(hits[session][rec_num], ci='sd')
    spike_rate.per_trial_spike_rate(stim[session][rec_num], ci='sd', subplots=subplots)
    name = exp[session].name
    plt.suptitle(f'Session {name} - pyramidal - per-trial across-units firing rate (aligned to push)')
    utils.save(fig_dir / f'trial_spike_rate_PC_cued+stim_push_{duration}s_{name}.png')
