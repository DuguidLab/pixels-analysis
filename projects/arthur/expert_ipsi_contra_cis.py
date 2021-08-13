"""
Ready-made ipsi & contra confidence intervals df for trained mice.

usage:
from expert_ipsi_contra_cis import *
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pixels import Experiment
from pixels.behaviours.reach import ActionLabels, Events, Reach 
from pixtools import spike_rate, utils, rolling_bin

mice = [       
    "HFR25",
    #"HFR29",
]

exp = Experiment(
    mice,
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
	'~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

fig_dir = Path('~/duguidlab/visuomotor_control/AZ_notes/npx-plots/expert')

# select units
units = exp.select_units(
    min_depth=0,
    max_depth=1200,
    #min_spike_width=0.4,
    name="cortex0-1200",
)

# define start, step, & end of confidence interval bins
start = 0.000
step = 0.200
end = 1.000
increment = 0.100

# get confidence interval for left & right visual stim.
cis_left = rolling_bin.get_rolling_bins(
    exp=exp,
    units=units,
    al=ActionLabels.miss_left,
    ci_s=start,
    step=step,
    ci_e=end,
    bl_s=-1.000,
    bl_e=-0.050,
    increment=increment,
)
assert False
# side of the PPC recording
sides = [
	'left',
	'right',
]

# no ipsi_m2 in trained group
contra_m2_list = []
ipsi_ppc_list = []
contra_ppc_list = []

for session in range(len(exp)):
	# m2
	contra_m2_list.append(cis_left[session][0])
	# ppc
	if sides[session] == 'left':
		ipsi_ppc_list.append(cis_left[session][1])
	else:
		contra_ppc_list.append(cis_left[session][1])

contra_m2_ci = pd.concat(
	contra_m2_list, axis=1, copy=False,
	keys=range(len(contra_m2_list)),
	names=['session', 'unit', 'bin']
)
ipsi_ppc_ci = pd.concat(
	ipsi_ppc_list, axis=1, copy=False,
	keys=range(len(ipsi_ppc_list)),
	names=['session', 'unit', 'bin']
	)
contra_ppc_ci = pd.concat(
	contra_ppc_list, axis=1, copy=False,
	keys=range(len(contra_ppc_list)),
	names=['session', 'unit', 'bin']
)
