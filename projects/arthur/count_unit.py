from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from pixels import Experiment
from pixels.behaviours.reach import VisualOnly, ActionLabels, Events
from pixtools import spike_rate, utils


mice = [       
    #'HFR19',
    'HFR20',
    #'HFR21',  # poor quality session
    'HFR22',
    'HFR23',
]

exp = Experiment(
    mice,
    VisualOnly,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

sns.set(font_scale=1)
fig_dir = Path('~/duguidlab/visuomotor_control/AZ_notes')

# Envelope for plots
ci = "sd"

## Select units

rec_num = 0
duration = 2

units = exp.select_units(
    min_depth=200,
    max_depth=1200,
    #min_spike_width=0.4,
    name="cortex200-1200",
)

## Spike rate plots for all visual stimulations

resps = exp.get_aligned_spike_rate_CI(
    ActionLabels.naive_left | ActionLabels.naive_right,
    Events.led_on,
    start=0,
    step=0.250,
    end=1,
    bl_start=-0.300,
    bl_end=-0.050,
    units=units,
)

data = []
areas = ["M2", "PPC"]

as_proportions = True

for session in range(len(exp)):
    name = exp[session].name

    for i, area in enumerate(areas):
        rec_resps = resps[session][i]
        units = rec_resps.columns.get_level_values('unit').unique()
        count = len(units)
        count_resp = 0

        for unit in units:
            cis = rec_resps[unit]
            for bin in cis:
                ci_bin = cis[bin]
                if ci_bin[2.5] > 0:
                    count_resp += 1
                    break
                elif ci_bin[97.5] < 0:
                    count_resp += 1
                    break

        if as_proportions:
            data.append((count_resp / count, area, name))
        else:
            data.append((count_resp, area, name))


if as_proportions:
    df = pd.DataFrame(data, columns=["Proportion of Neurons", "Brain Area", "Session"])
    sns.boxplot(data=df, x="Brain Area", y="Proportion of Neurons")
    sns.stripplot(data=df, x="Brain Area", y="Proportion of Neurons", color=".25", jitter=0)
    utils.save(fig_dir / f'Proportion_of_Neurons')

else:
    df = pd.DataFrame(data, columns=["Number of Neurons", "Brain Area", "Session"])
    sns.pointplot(data=df, x="Session", hue="Brain Area", y="Number of Neurons", dodge=True)
    utils.save(fig_dir / f'Number_of_Neurons')
