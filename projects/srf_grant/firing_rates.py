from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_rate, spike_times, utils

# Set to False to plot rasters instead
rates_not_rasters = True

duration = 2

# Sessions - targets
ses_m1_mth = 0
m1 = 0  # session 0
mth = 1  # session 0
ses_ipn_gpi = 1
ipn = 0  # session 1
gpi = 1  # session 1


mice = [       
    #'C57_1343253',  # has no behaviour
    'C57_1343255',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures/srf_grant')

if rates_not_rasters:
    firing_rates = exp.align_trials(
        ActionLabels.rewarded_push,
        Events.back_sensor_open,
        'spike_rate',
        duration=duration,
    )

    # IPN
    ses = exp[ses_ipn_gpi].name
    spike_rate.per_unit_spike_rate(firing_rates[ses_ipn_gpi][ipn], ci='sd')
    plt.suptitle(f'IPN - per-unit across-trials firing rate (aligned to push)')
    utils.save(fig_dir / f'IPN_unit_spike_rate_{duration}s_{ses}.png')

    # GPi
    spike_rate.per_unit_spike_rate(firing_rates[ses_ipn_gpi][gpi], ci='sd')
    plt.suptitle(f'GPi - per-unit across-trials firing rate (aligned to push)')
    utils.save(fig_dir / f'GPi_unit_spike_rate_{duration}s_{ses}.png')

    # MTh
    ses = exp[ses_m1_mth].name
    spike_rate.per_unit_spike_rate(firing_rates[ses_m1_mth][mth], ci='sd')
    plt.suptitle(f'Motor thalamus - per-unit across-trials firing rate (aligned to push)')
    utils.save(fig_dir / f'MTh_unit_spike_rate_{duration}s_{ses}.png')

else:
    times = exp.align_trials(
        ActionLabels.rewarded_push,
        Events.back_sensor_open,
        'spike_times',
        duration=duration,
    )

    # Plot rasters
    # MTh
    ses = exp[ses_m1_mth].name
    spike_times.per_unit_raster(times[ses_m1_mth][mth])
    plt.suptitle(f'Motor thalamus - per-unit across-trials rasters (aligned to push)')
    utils.save(fig_dir / f'MTh_unit_raster{duration}s_{ses}.png')

    # IPN
    ses = exp[ses_ipn_gpi].name
    spike_times.per_unit_raster(times[ses_ipn_gpi][ipn])
    plt.suptitle(f'IPN - per-unit across-trials rasters (aligned to push)')
    utils.save(fig_dir / f'IPN_unit_raster{duration}s_{ses}.png')

    # GPi
    spike_times.per_unit_raster(times[ses_ipn_gpi][gpi])
    plt.suptitle(f'GPi - per-unit across-trials rasters (aligned to push)')
    utils.save(fig_dir / f'GPi_unit_raster{duration}s_{ses}.png')
