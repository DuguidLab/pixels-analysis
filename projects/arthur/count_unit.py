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

area = ["M2", "PPC"][rec_num]

## Spike rate plots for all visual stimulations

#stim_all = exp.align_trials(
#    ActionLabels.naive_left | ActionLabels.naive_right,
#    Events.led_on,
#    'spike_rate',
#    duration=duration,
#    units=units,
#)

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
assert 0

data = []

for session in range(len(exp)):
    name = exp[session].name

    m2 = stim_all[session][0]
    ppc = stim_all[session][1]

    m2_resps = resps[session][0]
    ppc_resps = resps[session][1]

    count_m2 = len(m2.columns.get_level_values('unit').unique())
    count_m2_resp = 0
    for unit in m2.columns.get_level_values('unit').unique():
        cis = m2_resps[unit]
        if cis[2.5] > 0:
            count_m2_resp += 1
        elif cis[97.5] < 0:
            count_m2_resp += 1

    count_ppc = len(ppc.columns.get_level_values('unit').unique())
    count_ppc_resp = 0
    for unit in ppc.columns.get_level_values('unit').unique():
        cis = ppc_resps[unit]
        if cis[2.5] > 0:
            count_ppc_resp += 1
        elif cis[97.5] < 0:
            count_ppc_resp += 1

    print(f"{exp[session].name}: M2: {count_m2_resp}/{count_m2}, PPC: {count_ppc_resp}/{count_ppc}")
  
df = pd.DataFrame(data, columns=["Number of Neurons", "Brain Area", "Session"])

sns.pointplot(data=df, x="Session", hue="Brain Area", y="Number of Neurons", dodge=True)
utils.save(fig_dir / f'Number_of_Neurons')


