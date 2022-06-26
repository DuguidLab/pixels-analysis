#This file shall split 211024_VR46 into two groups by depth, then plot spike firing rate by unit
#First import required packages
import sys
import matplotlib.pyplot as plt


from matplotlib.pyplot import figure
from distutils.command.sdist import sdist
from base import *
from textwrap import wrap
sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis") 
from pixtools import spike_rate



#Now select the specific recording to be analysed
ses = myexp.get_session_by_name("211024_VR46")

#And now split session into two groups 
shallow = ses.select_units(
    group="good",
    max_depth=500,
    uncurated=False,
    name="shallow"
)

deep = ses.select_units(
    group="good",
    min_depth=550,
    max_depth=1200,
    uncurated=False,
    name="deep"
)

#And then define the hits
hits_s = ses.align_trials(
    ActionLabels.correct,
    Events.led_on,
    "spike_rate",
    duration=4,
    units=shallow
)

hits_d = ses.align_trials(
    ActionLabels.correct,
    Events.led_on,
    "spike_rate",
    duration=4,
    units=deep
)

#Using these units we may now plot seperate spike rate graphs
#First for shallow

name=ses.name
ci = 95

spike_rate.per_unit_spike_rate(hits_s, ci=ci)
plt.suptitle(
    "\n".join(wrap(f"Session {name} - per-unit (<500 Depth) across-trials firing rate (aligned to LED on)")),
)


#And then for deep units

spike_rate.per_unit_spike_rate(hits_d, ci=ci)
plt.suptitle(
    "\n".join(wrap(f"Session {name} - per-unit (500 < Depth < 1200) across-trials firing rate (aligned to LED on)")),
)


plt.show()
