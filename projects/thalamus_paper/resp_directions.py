import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
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

align_args = {
    "duration": 4,
    "min_depth": 500,
    "max_depth": 1200,
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

stim_miss = exp.align_trials(
    ActionLabels.uncued_laser_nopush,
    Events.laser_onset,
    'spike_rate',
    **align_args,
)

baseline = exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.tone_onset,
    'spike_rate',
    **align_args,
)

fig, axes = plt.subplots(len(exp), 2)

for session in range(len(exp)):
    pre_cue = baseline[session].loc[-499:0].mean()
    peri_push = hits[session].loc[-249:250].mean()
    cue_resp = peri_push - pre_cue

    pre_stim = stim[session].loc[-499:0].mean()
    post_stim = stim[session].loc[1:500].mean()
    stim_resp = post_stim - pre_stim

    pre_stim_miss = stim_miss[session].loc[-499:0].mean()
    post_stim_miss = stim_miss[session].loc[1:500].mean()
    stim_miss_resp = post_stim_miss - pre_stim_miss

    cue_responsives_pos = set()
    cue_responsives_neg = set()
    stim_responsives_pos = set()
    stim_responsives_neg = set()
    stim_miss_responsives_pos = set()
    stim_miss_responsives_neg = set()
    non_responsives = set()

    for unit in cue_resp.index.get_level_values('unit').unique():
        resp = False
        res = responsiveness.significant_CI(cue_resp[unit])
        if res:
            resp = True
            if res > 0:
                cue_responsives_pos.add(unit)
            else:
                cue_responsives_neg.add(unit)
        res = responsiveness.significant_CI(stim_resp[unit])
        if res:
            resp = True
            if res > 0:
                stim_responsives_pos.add(unit)
            else:
                stim_responsives_neg.add(unit)
        res = responsiveness.significant_CI(stim_miss_resp[unit])
        if res:
            resp = True
            if res > 0:
                stim_miss_responsives_pos.add(unit)
            else:
                stim_miss_responsives_neg.add(unit)
        if not resp:
            non_responsives.add(unit)

    venn3(
        [cue_responsives_pos, stim_responsives_pos, stim_miss_responsives_pos],
        ("Cued pushes", "Stim pushes", "Stim misses"),
        ax=axes[session][0]
    )
    venn3(
        [cue_responsives_neg, stim_responsives_neg, stim_miss_responsives_neg],
        ("Cued pushes", "Stim pushes", "Stim misses"),
        ax=axes[session][1]
    )
    axes[session][0].set_title(f"{exp[session].name} - positive")
    axes[session][1].set_title(f"{exp[session].name} - negative")
    plt.text(0.05, 0.95, len(non_responsives), transform=axes[session].transAxes)

save(f'Cued_vs_stim_push_vs_stim_miss_resp_pops_-_directions.png')
