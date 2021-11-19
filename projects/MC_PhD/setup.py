from pathlib import Path

from pixels import Experiment
from pixels.behaviours.reach import Reach, ActionLabels, Events

fig_dir = Path('~/duguidlab/visuomotor_control/figures')

mice = [
    #"VR37",
    #"VR40",
    #"VR42",
    #"VR50",
    #"VR46",
    "VR47",
    "VR49",
]

exp = Experiment(
    mice,
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)
