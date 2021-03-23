import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
fig_dir = Path('~/duguidlab/visuomotor_control/figures').expanduser()

def save(name):
    sns.set(font_scale=0.4)
    plt.gcf().savefig(fig_dir / name, bbox_inches='tight', dpi=300)

align_args = {
    "duration": 1,
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

widths = exp.get_spike_widths(
    min_depth=500,
    max_depth=1200,
)

data = [
    pd.DataFrame(  # spike widths < 0.4 ms
        [],
        columns=['Session', 'Response', 'Proportion']
    ),
    pd.DataFrame(  # spike widths >= 0.4 ms
        [],
        columns=['Session', 'Response', 'Proportion']
    ),
]


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

    # inner lists are for < 0.4 ms widths and >= 0.4 ms widths respectively
    cue_responsives_pos = [[], []]
    cue_responsives_neg = [[], []]
    stim_responsives_pos = [[], []]
    stim_responsives_neg = [[], []]
    #non_responsives = [[], []]

    units = cue_resp.index.get_level_values('unit').unique()
    ses_widths = widths[session]

    for unit in units:
        width = ses_widths[ses_widths['unit'] == unit]['median_ms']
        assert len(width.values) == 1
        if width.values[0] >= 0.4:
            cell_type = 1
        else:
            cell_type = 0

        resp = False
        res = responsiveness.significant_CI(cue_resp[unit])
        if res:
            resp = True
            if res == 1:
                cue_responsives_pos[cell_type].append(unit)
            else:
                cue_responsives_neg[cell_type].append(unit)
        res = responsiveness.significant_CI(stim_resp[unit])
        if res:
            resp = True
            if res == 1:
                stim_responsives_pos[cell_type].append(unit)
            else:
                stim_responsives_neg[cell_type].append(unit)
        #if not resp:
        #    non_responsives[cell_type].append(unit)

    for cell_type in (0, 1):
        active = len(cue_responsives_pos[cell_type]) + len(cue_responsives_neg[cell_type])
        summary =  [
            ('cue +ve', len(cue_responsives_pos[cell_type]) / active),
            ('cue -ve', len(cue_responsives_neg[cell_type]) / active),
            ('stim +ve', len(stim_responsives_pos[cell_type]) / active),
            ('stim -ve', len(stim_responsives_neg[cell_type]) / active),
            #('none', len(non_responsives) / num_units),
        ]

        for_df = []
        for response, proportion in summary:
            for_df.append({
                'Session': session,
                'Response': response,
                'Proportion': proportion,
            })
        data[cell_type] = data[cell_type].append(pd.DataFrame(for_df))


fig, axes = plt.subplots(2, 1, sharex=True)

sns.barplot(
    x="Session",
    y="Proportion",
    hue="Response",
    data=data[0],
    ax=axes[0]
)
axes[0].set_title('Median spike width < 0.4 ms')

sns.barplot(
    x="Session",
    y="Proportion",
    hue="Response",
    data=data[1],
    ax=axes[1]
)
axes[1].set_title('Median spike width >= 0.4 ms')

plt.suptitle("Cued vs stim modulated populations as proportion of cue-active")
save(f'cued_vs_stim_resp_pops_proportions.png')
