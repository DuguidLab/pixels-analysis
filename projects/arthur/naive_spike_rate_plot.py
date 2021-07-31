"""
Plot spike rate traces per unit, aligning to ipsi & contra visual stimualtion.
"""
from pathlib import Path
from naive_ipsi_contra_spike_rate import *

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns

for session in range(len(exp)):
    # per unit
    print(exp[session].name)

    subplots = spike_rate.per_unit_spike_rate(ipsi_m2[session], ci=ci)
    name = exp[session].name
    spike_rate.per_unit_spike_rate(contra_m2[session], ci=ci, subplots=subplots)
    name = exp[session].name
    plt.suptitle(f'Session {name} - m2-unit across-trials firing rate (aligned to ipsi & contra visual stimulation)')
    plt.gcf().set_size_inches(20, 20)
    utils.save(fig_dir / f'unit_spike_rate_ipsi&contra_visual_stim_{duration}s_m2_{name}', nosize=True)

    subplots = spike_rate.per_unit_spike_rate(ipsi_ppc[session], ci=ci)
    name = exp[session].name
    spike_rate.per_unit_spike_rate(contra_ppc[session], ci=ci, subplots=subplots)
    name = exp[session].name
    plt.suptitle(f'Session {name} - ppc-unit across-trials firing rate (aligned to ipsi & contra visual stimulation)')
    plt.gcf().set_size_inches(20, 20)
    utils.save(fig_dir / f'unit_spike_rate_ipsi&contra_visual_stim_{duration}s_ppc_{name}', nosize=True)

    # per trial
    # TODO
