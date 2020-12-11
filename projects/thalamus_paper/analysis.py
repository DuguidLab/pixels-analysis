import os

import matplotlib.pyplot as plt
import seaborn as sns
from spikesorters import Kilosort2Sorter

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events


HOME = os.path.expanduser('~')
Kilosort2Sorter.set_kilosort2_path(HOME + "/git/Kilosort2")

mice = [
    #"MCos5",
    #"MCos9",
    "MCos29",
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)


exp.sort_spikes()
