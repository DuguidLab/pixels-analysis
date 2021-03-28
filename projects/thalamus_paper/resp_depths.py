from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_rate, responsivenes, utils


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

stim_miss_depth = [stim_miss]
stim_miss_depth.append(exp.align_trials(
    ActionLabels.uncued_laser_nopush,
    Events.laser_onset,
    'spike_rate',
    duration=duration,
    min_depth=500,
    max_depth=1200,
))
stim_miss_depth.append(exp.align_trials(
    ActionLabels.uncued_laser_nopush,
    Events.laser_onset,
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
depths = ["all", "500 - 1200", "500 - 900"]


for depth in [0, 1, 2]:
    for session in range(len(exp)):
        pre_cue = baseline_depth[depth][session].loc[-499:0].mean()
        peri_push = hits_depth[depth][session].loc[-249:250].mean()
        cue_resp = peri_push - pre_cue

        pre_stim = stim_depth[depth][session].loc[-499:0].mean()
        post_stim = stim_depth[depth][session].loc[1:500].mean()
        stim_resp = post_stim - pre_stim

        pre_stim_miss = stim_miss_depth[depth][session].loc[-499:0].mean()
        post_stim_miss = stim_miss_depth[depth][session].loc[1:500].mean()
        stim_miss_resp = post_stim_miss - pre_stim_miss

        cue_responsives = set()
        stim_responsives = set()
        stim_miss_responsives = set()
        non_responsives = set()

        for unit in cue_resp.index.get_level_values('unit').unique():
            resp = False
            if responsiveness.significant_CI(cue_resp[unit]):
                resp = True
                cue_responsives.add(unit)
            if responsiveness.significant_CI(stim_resp[unit]):
                resp = True
                stim_responsives.add(unit)
            if responsiveness.significant_CI(stim_miss_resp[unit]):
                resp = True
                stim_miss_responsives.add(unit)
            if not resp:
                non_responsives.add(unit)

        venn3(
            [cue_responsives, stim_responsives, stim_miss_responsives],
            ("Cued pushes", "Stim pushes", "Stim misses"),
            ax=axes[session][depth]
        )
        axes[session][depth].set_title(f"{exp[session].name} - {depths[depth]}")
        plt.text(0.05, 0.95, len(non_responsives), transform=axes[session][depth].transAxes)

utils.save(fig_dir / f'Cued vs stim push vs stim miss resp pops, different depths.png')
