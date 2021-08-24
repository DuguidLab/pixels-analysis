from pathlib import Path

from pixels import Experiment
from pixels.behaviours.pushpull import PushPull

from preprocessing import exp

fig_dir = Path('~/duguidlab/Direction_Sensitivity/neuropixels_figures')

rec_num = 0

units = exp.select_units(
    min_depth=550,
    max_depth=900,
    name="550-900",
)

pyramidals = exp.select_units(
    min_depth=550,
    max_depth=900,
    min_spike_width=0.4,
    name="550-900-pyramidals",
)

interneurons = exp.select_units(
    min_depth=550,
    max_depth=900,
    max_spike_width=0.35,
    name="550-900-interneurons",
)
