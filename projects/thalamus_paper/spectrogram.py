import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_times


mice = [
    'C57_724',
    #'C57_1288723',
    #'C57_1288727',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

exp.set_cache(True)
sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures').expanduser()

duration = 2
hits = exp.align_trials(
    ActionLabels.rewarded_push,
    Events.back_sensor_open,
    'spike',
    duration=duration,
    raw=True,
)

fig, axes = plt.subplots(2, 2)
for i, ai in enumerate(axes):
    for j, aj in enumerate(ai):
        aj.specgram(hits[0][i][j], Fs=30000, NFFT=256*8)
        aj.set_ylim([0, 100])
plt.show()
