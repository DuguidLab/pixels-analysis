import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_times


mice = [
    'C57_724',
    'C57_1288723',
    'C57_1288727',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

exp.set_cache(False)
sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures').expanduser()

def save(name):
    fig.savefig(fig_dir / name, bbox_inches='tight', dpi=300)

duration = 4
bin_ms = 100

#hits = exp.align_trials(
#    ActionLabels.cued_shutter_push_full,
#    Events.back_sensor_open,
#    'spike_times',
#    duration=duration,
#)
#
#for session in range(len(exp)):
#    fig = spike_times.across_trials_histogram(hits, session, bin_ms=bin_ms, duration=duration)
#    name = exp[session].name
#    plt.suptitle(f'Session {name} - per-unit across-trials spike times (aligned to cued push)')
#    save(f'unit_spike_histograms_hits_{duration}s_{name}_in_brain.png')
#
#for session in range(len(exp)):
#    fig = spike_times.across_units_histogram(hits, session, bin_ms=bin_ms, duration=duration)
#    name = exp[session].name
#    plt.suptitle(f'Session {name} - per-trial across-unit spike times (aligned to cued push)')
#    save(f'trial_spike_histograms_hits_{duration}s_{name}_in_brain.png')


#stim = exp.align_trials(
#    ActionLabels.uncued_laser_push_full,
#    Events.back_sensor_open,
#    'spike_times',
#    duration=duration,
#)
#
#for session in range(len(exp)):
#    fig = spike_times.across_trials_histogram(stim, session, bin_ms=bin_ms, duration=duration)
#    name = exp[session].name
#    plt.suptitle(f'Session {name} - per-unit across-trials spike times (aligned to stim push)')
#    save(f'unit_spike_histograms_stim_{duration}s_{name}_in_brain.png')
#
#for session in range(len(exp)):
#    fig = spike_times.across_units_histogram(stim, session, bin_ms=bin_ms, duration=duration)
#    name = exp[session].name
#    plt.suptitle(f'Session {name} - per-trial across-unit spike times (aligned to stim push)')
#    save(f'trial_spike_histograms_stim_{duration}s_{name}_in_brain.png')


stim_miss = exp.align_trials(
    ActionLabels.uncued_laser_nopush,
    Events.laser_onset,
    'spike_times',
    duration=duration,
)

for session in range(len(exp)):
    fig = spike_times.across_trials_histogram(stim_miss, session, bin_ms=bin_ms, duration=duration)
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials spike times (aligned to nopush stim)')
    save(f'unit_spike_histograms_stim-miss_{duration}s_{name}_in_brain.png')

for session in range(len(exp)):
    fig = spike_times.across_units_histogram(stim_miss, session, bin_ms=bin_ms, duration=duration)
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-unit spike times (aligned to nopush stim)')
    save(f'trial_spike_histograms_stim-miss_{duration}s_{name}_in_brain.png')
