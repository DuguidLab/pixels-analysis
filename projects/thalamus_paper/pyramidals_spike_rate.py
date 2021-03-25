import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
from pathlib import Path

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

fig_dir = Path('~/duguidlab/visuomotor_control/figures').expanduser()

sns.set(font_scale=0.4)
def save(name):
    plt.gcf().savefig(fig_dir / name, bbox_inches='tight', dpi=300)

duration = 4

align_args = {
    "duration": duration,
    "min_depth": 500,
    "max_depth": 1200,
    "min_spike_width": 0.4,
}

hits = exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    'spike_rate',
    **align_args,
)

stim = exp.align_trials(
    ActionLabels.uncued_laser_push_full,
    Events.back_sensor_open,
    'spike_rate',
    **align_args,
)

for session in range(len(exp)):
    # per unit
    fig = spike_rate.across_trials_plot(hits[session])
    name = exp[session].name
    plt.suptitle(f'Session {name} - pyramidal - per-unit across-trials firing rate (aligned to cued push)')
    save(f'unit_spike_rate_PC_cued_push_{duration}s_{name}.png')

    fig = spike_rate.across_trials_plot(stim[session])
    name = exp[session].name
    plt.suptitle(f'Session {name} - pyramidal - per-unit across-trials firing rate (aligned to stim push)')
    save(f'unit_spike_rate_PC_stim_push_{duration}s_{name}.png')

    # per trial
    fig = spike_rate.across_units_plot(hits[session])
    name = exp[session].name
    plt.suptitle(f'Session {name} - pyramidal - per-trial across-units firing rate (aligned to cued push)')
    save(f'trial_spike_rate_PC_cued_push_{duration}s_{name}.png')

    fig = spike_rate.across_units_plot(stim[session])
    name = exp[session].name
    plt.suptitle(f'Session {name} - pyramidal - per-trial across-units firing rate (aligned to stim push)')
    save(f'trial_spike_rate_PC_stim_push_{duration}s_{name}.png')
