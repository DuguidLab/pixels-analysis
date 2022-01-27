"""
This outputs some spike times for Julian, who wants to look at excitatory vs inhibitory
units in L5.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_times, utils
from pixtools.clusters import unit_depths

output = Path("~/duguidlab/thalamus_paper/Npx_data/julian_data").expanduser()

mice = [
    'C57_724',
    'C57_1288723',
    'C57_1288727',
    'C57_1313404',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

rec_num = 0

units = exp.select_units(
    min_depth=500,
    max_depth=1200,
    name="500-1200",
)

select = {
    "units": units,
    "duration": 4,
}

results = {}

#results["hits"] = exp.align_trials(
#    ActionLabels.cued_shutter_push_full,
#    Events.back_sensor_open,
#    'spike_times',
#    **select,
#)
#
#results["laser_push_full"] = exp.align_trials(
#    ActionLabels.uncued_laser_push_full,
#    Events.back_sensor_open,
#    'spike_times',
#    **select,
#)
#
#results["laser_push_partial"] = exp.align_trials(
#    ActionLabels.uncued_laser_push_partial,
#    Events.laser_onset,
#    'spike_times',
#    **select,
#)
#
#results["laser_nopush"] = exp.align_trials(
#    ActionLabels.uncued_laser_nopush,
#    Events.laser_onset,
#    'spike_times',
#    **select,
#)

#for trials, data in results.items():
#    data.to_csv(output / f"{trials}_spike_times.csv")

depths = unit_depths(exp)
depths.to_csv(output / "unit_depths.csv")

spike_widths = exp[0].get_spike_widths(units)
spike_widths.to_csv(output / "spike_widths.csv")
