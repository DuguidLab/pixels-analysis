"""
Plot ipsi & contralateral visual stimulation responsive units across all layers, with cell type specified.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib_venn import venn2

from pixels import Experiment
from pixels.behaviours.reach import ActionLabels, Events, VisualOnly, Reach
from pixtools import utils, spike_rate, clusters, rolling_bin

fig_dir = Path("~/duguidlab/visuomotor_control/AZ_notes/npx-plots/naive")
sns.set(font_scale=0.4)

rec_num = 1
mice = [
    "HFR20",  # naive
    "HFR22",  # naive
    "HFR23",  # naive
]

exp = Experiment(
    mice,
    VisualOnly,
    "~/duguidlab/visuomotor_control/neuropixels",
    "~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON",
)

# define start, step, & end of confidence interval bins
start = 0.000
step = 0.200
end = 1.000

# exp.set_cache(False)
# M2 superfical layer 0-200um
pyramidals = exp.select_units(
    min_depth=0,
    max_depth=1200,
    min_spike_width=0.4,
    name="0-1200-pyramidals",
)

interneurons = exp.select_units(
    min_depth=0,
    max_depth=1200,
    max_spike_width=0.35,
    name="0-1200-interneurons",
)

units = [pyramidals, interneurons]
cell_type_names = ["Pyramidal", "Interneuron"]
area = ["M2", "PPC"][rec_num]

# find neurons that are responsive to left/right (ipsi/contralateral) visual stim.
lefts = rolling_bin.get_rolling_bins(
    exp=exp,
    units=units,
    al=ActionLabels.naive_left,
    ci_s=start,
    step=step,
    ci_e=end,
    bl_s=-1.000,
    bl_e=-0.050,
    increment=0.100,
)

rights = rolling_bin.get_rolling_bins(
    exp=exp,
    units=units,
    al=ActionLabels.naive_right,
    ci_s=start,
    step=step,
    ci_e=end,
    bl_s=-1.000,
    bl_e=-0.050,
    increment=0.100,
)

cluster_info = exp.get_cluster_info()

fig, axes = plt.subplots(1, len(exp), sharey=True)
results = []
py_proportion_all = []
intn_proportion_all = []
py_proportion_resps = []
intn_proportion_resps = []

for session in range(len(exp)):
    u_ids0 = lefts[0][session][rec_num].columns.get_level_values("unit").unique()
    u_ids1 = rights[0][session][rec_num].columns.get_level_values("unit").unique()
    assert not any(u_ids0 - u_ids1)

    total_num = len(units)
    pos = [[], []]  # inner lists are left and right
    neg = [[], []]  # inner lists are left and right
    non_resps = set()

    for c, cell_type in enumerate(units):
        for unit in cell_type[session][rec_num]:
            resp = False
            resps = [
                lefts[c][session][rec_num][unit],
                rights[c][session][rec_num][unit],
            ]

            for action, t in enumerate(resps):
                for b in np.arange(0, 1000, 100):
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

    left_resps = set(pos[0] + neg[0])
    right_resps = set(pos[1] + neg[1])
    both_resps = left_resps.intersection(right_resps)
    left_resps = left_resps.difference(both_resps)
    right_resps = right_resps.difference(both_resps)

    info = cluster_info[session][rec_num]
    probe_depth = exp[session].get_probe_depth()[rec_num]
    info["real_depth"] = probe_depth - info["depth"]

    data = []
    for c, cell_type in enumerate(units):
        for unit in cell_type[session][rec_num]:
            depth = info.loc[info["id"] == unit]["real_depth"].values[0]
            if unit in both_resps:
                group = "both"
            elif unit in left_resps:
                group = "left"
            elif unit in right_resps:
                group = "right"
            else:
                group = "none"
            data.append((session, unit, depth, group, cell_type_names[c]))

    if data:
        df = pd.DataFrame(
            data, columns=["Session", "ID", "depth", "group", "cell type"]
        )
#        sns.stripplot(
#            x="group",
#            order=["none", "left", "right", "both"],
#            y="depth",
#            hue="cell type",
#            data=df,
#            ax=axes[session],
#            palette="Set2",
#        )
#        axes[session].set_ylim(1200, 0)
#        axes[session].set_title(exp[session].name)
#        if session > 0:
#            axes[session].get_legend().remove()
#
#        results.extend(data)
#
#plt.gcf().set_size_inches(6, 5)
#utils.save(fig_dir / f"{mice}_left&right_{area}_responsives_by_depth_rolling_bin.pdf", nosize=True)
#
#plt.clf()
#df = pd.DataFrame(results, columns=["Session", "ID", "depth", "group", "cell type"])
#sns.stripplot(
#    x="group",
#    order=["none", "left", "right", "both"],
#    y="depth",
#    hue="cell type",
#    data=df,
#    palette="Set2",
#)
#plt.gca().set_ylim(1200, 0)
#plt.gca().set_title("Pooled results")
#utils.save(fig_dir / f"pooled_left&right_{area}_responsives_by_depth_rolling_bin.pdf", nosize=True)
#
    occurrence_all = df.value_counts('cell type')
    py_all = occurrence_all[0] / sum(occurrence_all)
    intn_all = 1 - py_all

    py_proportion_all.append(py_all)
    intn_proportion_all.append(intn_all)

    resps_df = df.set_index('group').drop('none')
    results.append(resps_df)
    occurrence_resps = resps_df.value_counts('cell type')
    py_resps = occurrence_resps[0] / sum(occurrence_resps)
    intn_resps = 1 - py_resps

    py_proportion_resps.append(py_resps)
    intn_proportion_resps.append(intn_resps)

# do i need this resps only df???
results_df = pd.concat(results)
assert False
plt.clf()
name = exp[session].name

_, axes = plt.subplots(2, 1, sharey=True)
sns.boxplot(
    data=intn_proportion_all,
    ax=axes[0],
)

sns.stripplot(
    data=intn_proportion_all,
    color=".25",
    jitter=0,
    ax=axes[0],
)
sns.boxplot(
    data=intn_proportion_resps,
    ax=axes[1],
)

sns.stripplot(
    data=intn_proportion_resps,
    color=".25",
    jitter=0,
    ax=axes[1],
)
#axes[0].set_xlabel(f'Pyramidal Neurons in All Good Units from {area}')
#axes[1].set_xlabel(f'Pyramidal Neurons in All Responsive Units from {area}')
#utils.save(
#    fig_dir / f"left&right_visual_stim._{area}_py_proportion_rolling_bins.pdf", nosize=True
#)
#print(f'pyramidal proportion in all good units from {area}: ', py_proportion_all)
#print(f'pyramidal proportion in resps good units from {area}: ', py_proportion_resps)

print(f'interneurons proportion in all good units from {area}: ', intn_proportion_all)
print(f'interneurons proportion in resps good units from {area}: ', intn_proportion_resps)
axes[0].set_xlabel(f'Interneurons in All Good Units from {area}')
axes[1].set_xlabel(f'Interneurons in All Responsive Units from {area}')
plt.yticks(np.arange(0, 1, 0.1))
plt.ylabel('Proportion')
utils.save(
    fig_dir / f"left&right_visual_stim._{area}_intn_proportion_rolling_bins.pdf", nosize=True
)
