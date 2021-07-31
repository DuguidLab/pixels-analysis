"""
Plot responsive neurons (ipsi & contra visual stim.) neurons in trained mice
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from pixels import Experiment
from pixels.behaviours.reach import Reach, ActionLabels, Events
from pixtools import spike_rate, utils

mice = [       
    'HFR25', #trained
    #'HFR29', #trained
]   

exp = Experiment(
    mice,
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

sns.set(font_scale=1)
fig_dir = Path('~/duguidlab/visuomotor_control/AZ_notes/npx-plots/expert')

## Select units
units = exp.select_units(
    min_depth=0, #see depth profile plot layer
    max_depth=1200, 
    #min_spike_width=0.4, #pyramidals
    #max_spike_width=0.35, #interneurons
    name="cortex0-1200",
)

resps_left = exp.get_aligned_spike_rate_CI(
    ActionLabels.miss_left,
    Events.led_on,
    start=0,
    step=0.250,
    end=1,
    bl_start=-1,
    bl_end=-0.050,
    units=units,
)

sides = [
    'left',
    'right',
]
ipsi = []
contra = []
rec_num = 0
areas = ["M2", "PPC"][rec_num]
#areas = ["PPC"] # for HFR29, right PPC only

as_proportions = True
in_ipsi = True

for session in range(len(exp)):
    if rec_num = 1 and sides[session] == 'left':
        for i, area in enumerate(areas):
            rec_resps_ipsi = resps_left[session][i]
            units = rec_resps_ipsi.columns.get_level_values('unit').unique()
            count_ipsi = len(units)
            count_resp_ipsi = 0
            for unit in units:
                cis = rec_resps_ipsi[unit]
                for bin in cis:
                    ci_bin = cis[bin]
                    if ci_bin[2.5] > 0:
                        count_resp_ipsi += 1
                        break
                    elif ci_bin[97.5] < 0:
                        count_resp_ipsi += 1
                        break

            if as_proportions:
                ipsi.append((count_resp_ipsi / count_ipsi, count, area, name))
            else:
                ipsi.append((count_resp_ipsi, count_ipsi, area, name))
    else:
        for i, area in enumerate(areas):
            rec_resps_contra = resps_left[session][i]
            units = rec_resps_contra.columns.get_level_values('unit').unique()
            count_contra = len(units)
            count_resp_contra = 0

            for unit in units:
                cis = rec_resps_contra[unit]
                for bin in cis:
                    ci_bin = cis[bin]
                    if ci_bin[2.5] > 0:
                        count_resp_contra += 1
                        break
                    elif ci_bin[97.5] < 0:
                        count_resp_contra += 1
                        break

            if as_proportions:
                contra.append((count_resp_contra / count_contra, count, area, name))
            else:
                contra.append((count_resp_contra, count_contra, area, name))

#plot total number of neurons
ipsi_df = pd.DataFrame(ipsi, columns=["Number of Ipsi Responsive Neurons", "Total Number of Neurons", "Brain Area", "Session"])
contra_df = pd.DataFrame(contra, columns=["Number of Contra Responsive Neurons", "Total Number of Neurons", "Brain Area", "Session"])
#df = pd.concat([dfn, dft], axis=1, keys=["naive", "trained"])
#preparing for t-test
sns.boxplot(data=df, x="Brain Area", y="Total Number of Neurons")
sns.stripplot(data=df, x="Brain Area", y="Total Number of Neurons", color=".25", jitter=0)
utils.save(fig_dir / f'Total_Number_of_Neurons')
print(count)
assert False #TODO

#plot proportion of responsive neurons
if as_proportions:
    if in_ipsi:
    # fix plotting thing, plot ipsi & contra next to each other, use subplot thing, len(exp)
        name = exp[session].name
        df = pd.DataFrame(ipsi, columns=["Proportion of Ipsi Responsive Neurons", "Total Number of Neurons", "Brain Area", "Session"])
        sns.boxplot(data=df, x="Brain Area", y="Proportion Ipsi of Responsive Neurons")
        sns.stripplot(data=df, x="Brain Area", y="Proportion of Ipsi Responsive Neurons", color=".25", jitter=0)
        utils.save(fig_dir / f'Proportion_of_Ipsi_Responsive_Neurons_{name}')
    else:
        name = exp[session].name
        df = pd.DataFrame(contra, columns=["Proportion of Contra Responsive Neurons", "Total Number of Neurons", "Brain Area", "Session"])
        sns.boxplot(data=df, x="Brain Area", y="Proportion Contra of Responsive Neurons")
        sns.stripplot(data=df, x="Brain Area", y="Proportion of Contra Responsive Neurons", color=".25", jitter=0)
        utils.save(fig_dir / f'Proportion_of_Contra_Responsive_Neurons_{name}')

#plot number of responsive neurons
else:
    if in_ipsi:
        df = pd.DataFrame(ipsi, columns=["Number of Ipsi Responsive Neurons", "Total Number of Neurons", "Brain Area", "Session"])
        sns.boxplot(data=df, x="Brain Area", y="Number of Ipsi Responsive Neurons")
        sns.stripplot(data=df, x="Brain Area", y="Number of Ipsi Responsive Neurons", color=".25", jitter=0)
        utils.save(fig_dir / f'Number_of_Ipsi_Responsive_Neurons_naive_{name}')
    else:
        df = pd.DataFrame(contra, columns=["Number of Contra Responsive Neurons", "Total Number of Neurons", "Brain Area", "Session"])
        sns.boxplot(data=df, x="Brain Area", y="Number of Contra Responsive Neurons")
        sns.stripplot(data=df, x="Brain Area", y="Number of Contra Responsive Neurons", color=".25", jitter=0)
        utils.save(fig_dir / f'Number_of_Contra_Responsive_Neurons_naive_{name}')
