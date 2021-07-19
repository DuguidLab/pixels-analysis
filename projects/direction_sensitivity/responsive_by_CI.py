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
    #"C57_1350953",
    "C57_1350954",
]

exp = Experiment(
    mice,
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
)

rec_num = 0
duration = 4
sns.set_context("paper")

units = exp.select_units(
    min_depth=550,
    max_depth=900,
    name="550-900",
)

pushes = exp.get_aligned_spike_rate_CI(
    ActionLabels.rewarded_push_good_mi,
    Events.motion_index_onset,
    start=-0.200,
    step=0.200,
    end=0.600,
    bl_event=Events.tone_onset,
    bl_start=-0.200,
    units=units,
)

pulls = exp.get_aligned_spike_rate_CI(
    ActionLabels.rewarded_pull_good_mi,
    Events.motion_index_onset,
    start=-0.200,
    step=0.200,
    end=0.600,
    bl_event=Events.tone_onset,
    bl_start=-0.200,
    units=units,
)

fig, axes = plt.subplots(1, len(exp))
plt.tight_layout()
results = {}


for session in range(len(exp)):
    units = pushes[session][rec_num].columns.get_level_values('unit').unique()
    units2 = pulls[session][rec_num].columns.get_level_values('unit').unique()
    assert not any(units - units2)

    pos = [[], []]  # inner lists are push then pull
    neg = [[], []]  # inner lists are push then pull
    non_resps = set()

    for unit in units:
        resp = False
        resps = [pushes[session][rec_num][unit], pulls[session][rec_num][unit]]

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

    venn2(
        [push_resps, pull_resps],
        ("Pushes", "Pulls"),
        ax=axes[session]
    )
    axes[session].set_title(exp[session].name)
    axes[session].text(0.05, 0.95, len(non_resps))


utils.save(fig_dir / f'push_pull_responsive_sizes.pdf')
