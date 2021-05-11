from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.reach import VisualOnly
from pixtools import clusters, utils

mice = [       
 ##  'HFR19',
    'HFR20',
 ##  'HFR21',
 ##  'HFR22',
 ##  'HFR23',
]

exp = Experiment(
    mice,
    VisualOnly,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/AZ_notes')

exp[0].draw_motion_index_rois()
exp[0].process_motion_index()
