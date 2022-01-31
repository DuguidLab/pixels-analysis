from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.reach import Reach, ActionLabels, Events
from pixtools import utils

fig_dir = Path('~/duguidlab/visuomotor_control/figures')
plt.tight_layout()

# TODO: Filter trials using DLC data if:
# - They are well-timed uncued reaches
# - The mouse hasn't stopped repetitive reaching since the previous trial

# TODO: Pool across days for same mouse?

mice = [
    # COHORT 1
    #"VR37",  # Mostly noise
    #"VR40",  # Mostly noise, poor behaviour
    #"VR42",  # Less noisy, alright behaviour - to consider

    # COHORT 2
    "VR46",
        # 211024_VR46: Good, TODO: Crop action labels to avoid noise at end
        # 211025_VR46: Good, crop out noisy parts. MISALIGNED?
    "VR47",
        # 211030_VR47: Good
        # 211031_VR47: Crop out noisy parts. Maybe find optimal window if d' low
        # 211101_VR47: Good
    "VR49",
        # 211027_VR49: Good, first or second segment may be bad
        # 211028_VR49: Good
    "VR50",
        # 211012_VR50: Good

    # COHORT 3
]

exp = Experiment(
    mice,
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

## Depths
"""
VR46:
 - Implanted to 3.6 mm
 - entry at (0.3, 0.8, 0)
 - leaves cortex at (0.8, 0.32, 1)
 - terminates at (1.3, -0.22, 3.25)
 - distance implanted ~3.55
 - M2 ~ 0.2 - 1.2 mm

VR47:
 - Implanted to 2.5 mm
 - entry at (0.4, 0.86, 0)
 - leaves cortex at (0.8, 0.5, 1.1)
 - APPEARS TO terminate at (0.9, 0.38, 1.6) but enters ventricle
 - which would mean distance implanted ~1.74
 - M2 ~ 0.1 - 1.22 mm
 - Assumed depth of 2.5 mm - cross-reference with good units

VR49:
 - Implanted to 3.6 mm
 - entry at (0.7, 0.32, 0)
 - leaves cortex at (0.85, 0.28, 1.25)
 - terminates at (1.4, -0.34, 3.5)
 - distance implanted ~3.63
 - M2 ~ 0.1 - 1.25 mm

VR50:
 - Implanted to 3.4 mm
 - entry at (0.5, 0.35, 0)
 - leaves cortex at (0.8, 0.08, 1.25)
 - terminates at (1.3, -0.36, 3.1)
 - distance implanted ~3.3
 - M2 ~ 0.15 - 1.3 mm

"""

units = exp.select_units(
    max_depth=1500,
    name="up_to_1500",
)
