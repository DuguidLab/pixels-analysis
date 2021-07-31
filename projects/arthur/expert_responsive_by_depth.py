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
from pixels.behaviours.reach import ActionLabels, Events, Reach
from pixtools import utils, spike_rate, clusters

fig_dir = Path("~/duguidlab/visuomotor_control/AZ_notes/npx-plots/expert")
sns.set(font_scale=0.4)

mice = [
    "HFR25",  # trained
    'HFR29', #trained, right PPC only
]

exp = Experiment(
    mice,
    Reach,
    "~/duguidlab/visuomotor_control/neuropixels",
    "~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON",
)

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

units = [pyramidals, interneurons]
cell_type_names = ["Pyramidal", "Interneuron"]
rec_num = 0
area = ["M2", "PPC"][rec_num]
# area = ['right PPC'][rec_num]

# find neurons that are responsive to left/right (ipsi/contralateral) visual stim.
# trained, only left visual stim.
lefts = [
    exp.get_aligned_spike_rate_CI(
        ActionLabels.miss_left,
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

cluster_info = exp.get_cluster_info()

fig, axes = plt.subplots(1, len(exp), sharey=True)
if len(exp) == 1:
    axes = [axes]

for session in range(len(exp)):
    pos = [[]]
    neg = [[]]
    non_resps = set()

    data = []
    for c, cell_type in enumerate(units):
        for unit in cell_type[session][rec_num]:
            resp = False
            resps = [lefts[c][session][rec_num][unit]]

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

    info = cluster_info[session][rec_num]
    probe_depth = exp[session].get_probe_depth()[rec_num]
    info["real_depth"] = probe_depth - info["depth"]

    for c, cell_type in enumerate(units):
        for unit in cell_type[session][rec_num]:
            depth = info.loc[info["id"] == unit]["real_depth"].values[0]
            if unit in left_resps:
                group = "left"
            else:
                group = "none"
            data.append((unit, depth, group, cell_type_names[c]))

    df = pd.DataFrame(data, columns=["ID", "depth", "group", "cell type"])
    sns.stripplot(
        x="group",
        order=["none", "left"],
        y="depth",
        hue="cell type",
        data=df,
        ax=axes[session],
    )
    axes[session].set_ylim(1200, 0)
    axes[session].set_title(exp[session].name)
#       if session > 0:
#           axes[session].get_legend().remove()

plt.gcf().set_size_inches(6, 5)
utils.save(
    fig_dir / f"left_visual_stim._{area}_responsives_by_depth_{mice}.pdf", nosize=True
)
