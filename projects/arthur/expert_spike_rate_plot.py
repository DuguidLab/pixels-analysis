"""
Plot spike rate traces per unit, aligning to ipsi & contra visual stimualtion.
"""
from pathlib import Path
from expert_ipsi_contra_spike_rate import *

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns

## Select units
rec_num = 1

for session in range(len(exp)):
	fig = spike_rate.per_unit_spike_rate(stim_left[session][rec_num], ci=ci)
	name = exp[session].name
	plt.suptitle(f'Session {name} - unit across-trials firing rate (aligned to left visual stimulation)')
	plt.gcf().set_size_inches(20, 20)
	utils.save(fig_dir / f'unit_spike_rate_left_visual_stim_{duration}s_{name}_{rec_num}', nosize=True)

#   fig = spike_rate.per_unit_spike_rate(contra_m2[session], ci=ci)
#   name = exp[session].name
#   plt.suptitle(f'Session {name} - m2-unit across-trials firing rate (aligned to contra visual stimulation)')
#   plt.gcf().set_size_inches(20, 20)
#   utils.save(fig_dir / f'unit_spike_rate_contra_visual_stim_{duration}s_m2_{name}', nosize=True)

#    fig = spike_rate.per_unit_spike_rate(ipsi_ppc[session], ci=ci)
#    name = exp[session].name
#    plt.suptitle(f'Session {name} - ppc-unit across-trials firing rate (aligned to ipsi visual stimulation)')
#    plt.gcf().set_size_inches(20, 20)
#    utils.save(fig_dir / f'unit_spike_rate_ipsi_visual_stim_{duration}s_ppc_{name}', nosize=True)

#    fig = spike_rate.per_unit_spike_rate(contra_ppc[session], ci=ci)
#    name = exp[session].name
#    plt.suptitle(f'Session {name} - ppc-unit across-trials firing rate (aligned to ipsi & contra visual stimulation)')
#    plt.gcf().set_size_inches(20, 20)
#    utils.save(fig_dir / f'unit_spike_rate_contra_visual_stim_{duration}s_ppc_{name}', nosize=True)

    # per trial
    # TODO
