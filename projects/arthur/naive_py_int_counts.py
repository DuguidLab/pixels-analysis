"""
Plot proportion of responsive pyramidal & interneurons.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib_venn import venn2

from pixels import Experiment
from pixels.behaviours.reach import ActionLabels, Events, VisualOnly, Reach
from pixtools import utils, spike_rate, clusters

fig_dir = Path("~/duguidlab/visuomotor_control/AZ_notes/npx-plots/naive")
sns.set(font_scale=0.4)

mice = [
    #"HFR20",  # naive
    "HFR22",  # naive
    #"HFR23",  # naive
]

exp = Experiment(
    mice,
    VisualOnly,
    "~/duguidlab/visuomotor_control/neuropixels",
    "~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON",
)

rec_num = 1
duration = 2

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

assert False
units = [pyramidals, interneurons]
cell_type_names = ["Pyramidal", "Interneuron"]
area = ["M2", "PPC"][rec_num]

# find neurons that are responsive to left/right (ipsi/contralateral) visual stim.
# naive
lefts = [
    exp.get_aligned_spike_rate_CI(
        ActionLabels.naive_left,
        Events.led_on,
        start=0,
        step=0.200,
        end=1,
        bl_start=-1,
        bl_end=-0.050,
        units=u,
    )
    for u in units
]
lefts = [
    exp.get_aligned_spike_rate_CI(
        ActionLabels.naive_left,
        Events.led_on,
        start=0.050,
        step=0.200,
        end=1.050,
        bl_start=-1,
        bl_end=-0.050,
        units=u,
    )
    for u in units
]
rights = [
    exp.get_aligned_spike_rate_CI(
        ActionLabels.naive_right,
        Events.led_on,
        start=0,
        step=0.200,
        end=1,
        bl_start=-1,
        bl_end=-0.050,
        units=u,
    )
    for u in units
]

results = []

for session in range(len(exp)):
    u_ids1 = lefts[0][session][rec_num].columns.get_level_values("unit").unique()
    u_ids2 = rights[0][session][rec_num].columns.get_level_values("unit").unique()
    assert not any(u_ids1 - u_ids2)

    total_num = len(units)
    pos = [[], []]  # inner lists are left and right
    neg = [[], []]  # inner lists are left and right
    non_resps = set()
    pym = set()
    intn = set()

    for c, cell_type in enumerate(units):
        for unit in cell_type[session][rec_num]:
            resp = False
            resps = [
                lefts[c][session][rec_num][unit],
                rights[c][session][rec_num][unit],
            ]

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
        sns.stripplot(
            x="group",
            order=["none", "left", "right", "both"],
            y="depth",
            hue="cell type",
            data=df,
            ax=axes[session],
            palette="Set2",
        )
        axes[session].set_ylim(1200, 0)
        axes[session].set_title(exp[session].name)
        if session > 0:
            axes[session].get_legend().remove()

        results.extend(data)

plt.gcf().set_size_inches(6, 5)
utils.save(fig_dir / f"{mice}_left&right_{area}_responsives_by_depth.pdf", nosize=True)

plt.clf()
df = pd.DataFrame(results, columns=["Session", "ID", "depth", "group", "cell type"])
sns.stripplot(
    x="group",
    order=["none", "left", "right", "both"],
    y="depth",
    hue="cell type",
    data=df,
    palette="Set2",
)
plt.gca().set_ylim(1200, 0)
plt.gca().set_title("Pooled results")
utils.save(fig_dir / f"pooled_left&right_{area}_responsives_by_depth.pdf", nosize=True)
