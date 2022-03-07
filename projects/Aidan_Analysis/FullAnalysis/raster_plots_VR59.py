#First import experimental data from base.py
from cProfile import label

from matplotlib.pyplot import subplots
from base import *


#Now add the pixtools directory to the path to allow us to use the modules within
sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")
from pixtools.utils import Subplots2D


#Then import the data to allow raster plotting
from rasterbase import *

#Now let us plot the raster info for the 17th, per UNIT


unit1 = myexp.select_units(
    group="good",
    max_depth=1500,
    name="cortex"
)



#Now take all spike timees for all left hand events
session1_L = myexp.align_trials(
    ActionLabels.correct_left,
    Events.led_on,
    "spike_times",
    duration=4,
    units = unit1
)

#And now as right hand 
session1_R = myexp.align_trials(
    ActionLabels.correct_right,
    Events.led_on,
    "spike_times",
    duration=4,
    units = unit1
)


#Now plot left and right as subplots, remember we are taking session 2 here (by indexing into session)
subplot1 = per_unit_raster(session1_L[1], sample=None, start=0, subplots=None, label=True)

per_unit_raster(
    session1_R[1],
    sample=None,
    start=0,
    subplots=subplot1,
    label=True
)

plt.show()
