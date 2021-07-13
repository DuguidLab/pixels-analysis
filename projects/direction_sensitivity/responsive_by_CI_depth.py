from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib_venn import venn2

from pixels import Experiment
from pixels.behaviours.pushpull import ActionLabels, Events, PushPull
from pixtools import utils

fig_dir = Path('~/duguidlab/Direction_Sensitivity/neuropixels_figures')

mice = [       
    "C57_1350950",
    "C57_1350951",
    "C57_1350952",
    #"C57_1350953",  # MI done, needs curation
    "C57_1350954",
]

exp = Experiment(
    mice,
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
)

rec_num = 0
duration = 4

pyramidals = exp.select_units(
    min_depth=550,
    max_depth=900,
    min_spike_width=0.4,
    name="550-900-pyramidals",
)

interneurons = exp.select_units(
    min_depth=550,
    max_depth=900,
    max_spike_width=0.35,
    name="550-900-interneurons",
)

units = [pyramidals, interneurons]
cell_type_names = ["Pyramidal", "Interneuron"]

pushes = [
    exp.get_aligned_spike_rate_CI(
        ActionLabels.rewarded_push_good_mi,
        Events.motion_index_onset,
        start=-0.200,
        step=0.200,
        end=0.600,
        bl_event=Events.tone_onset,
        bl_start=-0.200,
        units=u,
    )
    for u in units
]

pulls = [
    exp.get_aligned_spike_rate_CI(
        ActionLabels.rewarded_pull_good_mi,
        Events.motion_index_onset,
        start=-0.200,
        step=0.200,
        end=0.600,
        bl_event=Events.tone_onset,
        bl_start=-0.200,
        units=u,
    )
    for u in units
]

cluster_info = exp.get_cluster_info()

fig, axes = plt.subplots(1, len(exp), sharey=True)
#plt.tight_layout()
results = {}


for session in range(len(exp)):
    units = pushes[session][rec_num].columns.get_level_values('unit').unique()
    units2 = pulls[session][rec_num].columns.get_level_values('unit').unique()
    assert not any(units - units2)

    pos = [[], []]  # inner lists are push then pull
    neg = [[], []]  # inner lists are push then pull
    non_resps = set()

    for c, cell_type in enumerate(units):
        for unit in cell_type:
            resp = False
            resps = [pushes[c][session][rec_num][unit], pulls[c][session][rec_num][unit]]

            for action, t in enumerate(resps):
                for b in range(4):
                    if 0 < t[b][2.5]:
                        pos[action].append(unit)
                        resp = True
                        break

                    elif t[b][97.5] < 0:
                        neg[action].append(unit)
                        resp = True
                        break

            if not resp:
                non_resps.add(unit)

    push_resps = set(pos[0] + neg[0])
    pull_resps = set(pos[1] + neg[1])
    both_resps = push_resps.intersection(pull_resps)
    push_resps = push_resps.difference(both_resps)
    pull_resps = pull_resps.difference(both_resps)

    info = cluster_info[session][rec_num]
    probe_depth = exp[session].get_probe_depth()[rec_num]
    info["real_depth"] = probe_depth - info["depth"]

    data = []
    for c, cell_type in enumerate(units):
        for unit in cell_type:
            depth = info.loc[info["id"] == unit]["real_depth"].values[0]
            if unit in both_resps:
                group = "both"
            elif unit in push_resps:
                group = "push"
            elif unit in pull_resps:
                group = "pull"
            else:
                group = "none"
            data.append((unit, depth, group, cell_type_names[c]))

    df = pd.DataFrame(data, columns=["ID", "depth", "group", "cell type"])
    sns.stripplot(
        x="group",
        y="depth",
        hue="cell type",
        data=df,
        ax=axes[session],
    )
    axes[session].set_aspect(0.01)
    axes[session].set_ylim(980, 520)

utils.save(fig_dir / f'push_pull_responsives_by_depth.pdf')
