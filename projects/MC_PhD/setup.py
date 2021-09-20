from pathlib import Path

from pixels import Experiment
from pixels.behaviours.reach import Reach

fig_dir = Path('~/duguidlab/visuomotor_control/figures')

mice = [
    "VR37",
    "VR40",
    "VR42",
]

exp = Experiment(
    mice,
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)
