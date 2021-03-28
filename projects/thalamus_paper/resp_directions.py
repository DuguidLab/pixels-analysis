"""
NOTE: THIS SCRIPT HAS BEEN REPLACED BY  "pyramidals_resp_cue_vs_stim.py" WHICH USES
"Experiment.get_aligned_spike_rate_CI".
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_rate, responsiveness, utils


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
rec_num = 0

select = {
    "duration": 0.5,
    "min_depth": 500,
    "max_depth": 1200,
    "min_spike_width": 0.4,
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

#stim_miss = exp.align_trials(
#    ActionLabels.uncued_laser_nopush,
#    Events.laser_onset,
#    'spike_rate',
#    **select,
#)

baseline = exp.align_trials(
    ActionLabels.cued_shutter_push_full,
    Events.tone_onset,
    'spike_rate',
    **select,
)

fig, axes = plt.subplots(len(exp), 2)

for session in range(len(exp)):
    #pre_cue = baseline[session][rec_num].loc[-199:0].mean()
    #peri_push = hits[session][rec_num].loc[-99:100].mean()
    pre_cue = baseline[session][rec_num].loc[-499:0].mean()
    peri_push = hits[session][rec_num].loc[-149:250].mean()
    cue_resp = peri_push - pre_cue

    #pre_stim = stim[session][rec_num].loc[-199:0].mean()  # TODO: doesn't make sense to baseline to pre-push
    #post_stim = stim[session][rec_num].loc[-99:100].mean()
    pre_stim = stim[session][rec_num].loc[-499:0].mean()
    post_stim = stim[session][rec_num].loc[-149:250].mean()
    stim_resp = post_stim - pre_stim

    #pre_stim_miss = stim_miss[session][rec_num].loc[-499:0].mean()
    #post_stim_miss = stim_miss[session][rec_num].loc[1:500].mean()
    #stim_miss_resp = post_stim_miss - pre_stim_miss

    cue_responsives_pos = set()
    cue_responsives_neg = set()
    stim_responsives_pos = set()
    stim_responsives_neg = set()
    #stim_miss_responsives_pos = set()
    #stim_miss_responsives_neg = set()
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
        #res = responsiveness.significant_CI(stim_miss_resp[unit])
        #if res:
        #    resp = True
        #    if res > 0:
        #        stim_miss_responsives_pos.add(unit)
        #    else:
        #        stim_miss_responsives_neg.add(unit)
        if not resp:
            non_responsives.add(unit)

    print(f"session: {exp[session].name}, {len(cue_responsives_pos)}, {len(cue_responsives_neg)}")
    venn2(
        [cue_responsives_pos, stim_responsives_pos],
        ("Cued pushes", "Stim pushes", "Stim misses"),
        ax=axes[session][0]
    )
    venn2(
        [cue_responsives_neg, stim_responsives_neg],
        ("Cued pushes", "Stim pushes"),
        ax=axes[session][0]
    )
    axes[session][0].set_title(f"{exp[session].name} - positive")
    axes[session][1].set_title(f"{exp[session].name} - negative")
    plt.text(0.05, 0.95, len(non_responsives), transform=axes[session][0].transAxes)

utils.save(fig_dir / f'Cued_vs_stim_push_vs_stim_miss_resp_pops_-_directions.png')
