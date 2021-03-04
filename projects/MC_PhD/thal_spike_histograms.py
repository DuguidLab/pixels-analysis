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

duration = 4
hits = exp.align_trials(
    ActionLabels.rewarded_push,
    Events.back_sensor_open,
    'spike_times',
    duration=duration,
)

bin_ms = 100

for session in range(len(exp)):
    fig = spike_times.across_trials_histogram(hits, session, bin_ms=bin_ms, duration=duration)
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-unit across-trials spike times (aligned to push)')
    fig.savefig(fig_dir / f'unit_spike_histograms_{duration}s_{name}_in_brain.png', bbox_inches='tight', dpi=300)

for session in range(len(exp)):
    fig = spike_times.across_units_histogram(hits, session, bin_ms=bin_ms, duration=duration)
    name = exp[session].name
    plt.suptitle(f'Session {name} - per-trial across-unit spike times (aligned to push)')
    fig.savefig(fig_dir / f'trial_spike_histograms_{duration}s_{name}_in_brain.png', bbox_inches='tight', dpi=300)
