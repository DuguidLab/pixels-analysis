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
from pixtools import spike_rate, utils

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

duration = 2

# select units
units = exp.select_units(
    min_depth=0,
    max_depth=1200,
    #min_spike_width=0.4,
    name="cortex0-1200",
)

# define start, step, & end of confidence interval bins
start = 0.000
step = 0.250
end = 1.000

# get confidence interval for left & right visual stim.
cis_left = exp.get_aligned_spike_rate_CI(
    ActionLabels.miss_left,
    Events.led_on,
    start=start,
    step=step,
    end=end,
    bl_start=-1.000,
    bl_end=0.000,
    units=units,
)

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
