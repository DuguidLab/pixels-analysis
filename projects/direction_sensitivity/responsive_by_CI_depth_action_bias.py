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

start = -0.250
step = 0.250
end = 1.500
#start = -0.200
#step = 0.200
#end = 1.400

pushes = [
    exp.get_aligned_spike_rate_CI(
        ActionLabels.rewarded_push_good_mi,
        Events.motion_index_onset,
        start=start,
        step=step,
        end=end,
        bl_event=Events.tone_onset,
        bl_start=start,
        units=u,
    )
    for u in units
]

pulls = [
    exp.get_aligned_spike_rate_CI(
        ActionLabels.rewarded_pull_good_mi,
        Events.motion_index_onset,
        start=start,
        step=step,
        end=end,
        bl_event=Events.tone_onset,
        bl_start=start,
        units=u,
    )
    for u in units
]

cluster_info = exp.get_cluster_info()

fig, axes = plt.subplots(1, len(exp), sharey=True)
results = {}

print("Session            none push pull push-bias pull-bias no-bias both-bias")

for session in range(len(exp)):
    u_ids1 = pushes[0][session][rec_num].columns.get_level_values('unit').unique()
    u_ids2 = pulls[0][session][rec_num].columns.get_level_values('unit').unique()
    assert not any(u_ids1 - u_ids2)

    push_only = set()
    pull_only = set()
    push_bias = set()
    pull_bias = set()
    no_bias = set()
    both_bias = set()
    non_resps = set()

    for c, cell_type in enumerate(units):
        for unit in cell_type[session][rec_num]:
            resp = False
            u_push = pushes[c][session][rec_num][unit]
            u_pull = pulls[c][session][rec_num][unit]

            if (0 < u_push.loc[2.5]).any() or (0 > u_push.loc[97.5]).any():
                resp = True
                push_only.add(unit)

            if (0 < u_pull.loc[2.5]).any() or (0 > u_pull.loc[97.5]).any():
                resp = True
                if unit in push_only:
                    push_only.remove(unit)
                    # check bias
                    b_push = b_pull = False
                    # We check both as both can be true
                    if (u_pull.loc[97.5] < u_push.loc[2.5]).any():
                        b_push = True
                    if (u_push.loc[97.5] < u_pull.loc[2.5]).any():
                        b_pull = True
                    if b_push and b_pull:
                        both_bias.add(unit)
                    elif b_push and not b_pull:
                        push_bias.add(unit)
                    elif b_pull and not b_push:
                        pull_bias.add(unit)
                    else:
                        no_bias.add(unit)
                else:
                    pull_only.add(unit)

            if not resp:
                non_resps.add(unit)

    info = cluster_info[session][rec_num]
    probe_depth = exp[session].get_probe_depth()[rec_num]
    info["real_depth"] = probe_depth - info["depth"]

    data = []
    for c, cell_type in enumerate(units):
        for unit in cell_type[session][rec_num]:
            depth = info.loc[info["id"] == unit]["real_depth"].values[0]
            if unit in push_only:
                group = "push_only"
            if unit in pull_only:
                group = "pull_only"
            elif unit in push_bias:
                group = "push_bias"
            elif unit in pull_bias:
                group = "pull_bias"
            elif unit in no_bias:
                group = "no_bias"
            elif unit in both_bias:
                group = "both_bias"
            else:
                group = "none"
            data.append((unit, depth, group, cell_type_names[c]))
    
    labels = ["none", "push_only", "pull_only", "push_bias", "pull_bias", "no_bias", "both_bias"]
    df = pd.DataFrame(data, columns=["ID", "depth", "group", "cell type"])
    sns.stripplot(
        x="group",
        order=labels,
        y="depth",
        hue="cell type",
        data=df,
        ax=axes[session],
    )
    axes[session].set_ylim(980, 520)
    if session > 0:
        axes[session].get_legend().remove()

    axes[session].set_xticklabels(labels, rotation=45);

    print(
        exp[session].name,
        len(non_resps), "    ", len(push_only), "    ", len(pull_only),
        "    ", len(push_bias), "    ", len(pull_bias), "    ", len(no_bias), "    ",
        len(both_bias)
    )

plt.gcf().set_size_inches(8, 5)
utils.save(fig_dir / f'push_pull_responsives_by_depth_and_bias_{step}.pdf', nosize=True)
