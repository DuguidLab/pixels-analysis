import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
from pathlib import Path

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_rate, responsiveness
from pixtools.utils import subplots2d


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

exp.set_cache(True)
sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures').expanduser()

def save(name):
    fig.savefig(fig_dir / name, bbox_inches='tight', dpi=300)

duration = 4


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

baseline = exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.tone_onset,
    'spike_rate',
    duration=duration,
)

for session in range(len(exp)):
    # per unit
    fig = spike_rate.across_trials_plot(hits[session])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to cued push)')
    save(f'unit_spike_rate_cued_push_{duration}s_{name}.png')

    fig = spike_rate.across_trials_plot(stim[session])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to stim push)')
    save(f'unit_spike_rate_stim_push_{duration}s_{name}.png')

    fig = spike_rate.across_trials_plot(stim_miss[session])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to nopush stim)')
    save(f'unit_spike_rate_stim-miss_{duration}s_{name}.png')

    # per trial
    fig = spike_rate.across_units_plot(hits[session])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to cued push)')
    save(f'trial_spike_rate_cued_push_{duration}s_{name}.png')

    fig = spike_rate.across_units_plot(stim[session])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to stim push)')
    save(f'trial_spike_rate_stim_push_{duration}s_{name}.png')

    fig = spike_rate.across_units_plot(stim_miss[session])
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to nopush stim)')
    save(f'trial_spike_rate_stim-miss_{duration}s_{name}.png')


hits_depth = [hits]
hits_depth.append(exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    'spike_rate',
    duration=duration,
    min_depth=500,
    max_depth=1200,
))
hits_depth.append(exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    'spike_rate',
    duration=duration,
    min_depth=500,
    max_depth=900,
))

stim_depth = [stim]
stim_depth.append(exp.align_trials(
    ActionLabels.uncued_laser_push_full,
    Events.back_sensor_open,
    'spike_rate',
    duration=duration,
    min_depth=500,
    max_depth=1200,
))
stim_depth.append(exp.align_trials(
    ActionLabels.uncued_laser_push_full,
    Events.back_sensor_open,
    'spike_rate',
    duration=duration,
    min_depth=500,
    max_depth=900,
))

baseline_depth = [baseline]
baseline_depth.append(exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.tone_onset,
    'spike_rate',
    duration=duration,
    min_depth=500,
    max_depth=1200,
))
baseline_depth.append(exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.tone_onset,
    'spike_rate',
    duration=duration,
    min_depth=500,
    max_depth=900,
))

## Responsive populations
fig, axes = plt.subplots(len(exp), 3)

for depth in [0, 1, 2]:
    for session in range(len(exp)):
        pre_cue = baseline_depth[depth][session].loc[-499:0].mean()
        peri_push = hits_depth[depth][session].loc[-149:150].mean()
        cue_resp = peri_push - pre_cue

        pre_stim = stim_depth[depth][session].loc[-499:0].mean()
        post_stim = stim_depth[depth][session].loc[1:500].mean()
        stim_resp = post_stim - pre_stim

        cue_responsives = set()
        stim_responsives = set()
        non_responsives = set()

        for unit in cue_resp.index.get_level_values('unit').unique():
            resp = False
            res = responsiveness.significant_CI(cue_resp[unit]):
                resp = True
                cue_responsives.add(unit)
            if responsiveness.significant_CI(stim_resp[unit]):
                resp = True
                stim_responsives.add(unit)
            if not resp:
                non_responsives.add(unit)

        venn2([cue_responsives, stim_responsives], ("Cued pushes", "Stim pushes"), ax=axes[session][depth])
        axes[session].set_title(exp[session].name)
        plt.text(0.05, 0.95, len(non_responsives), transform=axes[session][depth].transAxes)

save(f'Cued vs stim push resp pops 500ms bins, depths.png')
