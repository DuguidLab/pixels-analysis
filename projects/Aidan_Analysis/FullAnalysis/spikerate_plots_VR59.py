#Let us now plot the spike rates for the desired experiment, here aligned to the LED on
#Take all correct trials from both left and right sides
#Remember this all requires data be processed beforehand (use processing.py!)

from distutils.command.sdist import sdist
from base import *


sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis") #Adds the location of the pixtools folder to the path
from pixtools import spike_rate #Allows us to use the custom class designed to give the figs/axes from a set of subplots that will fit the data. This generates as many subplots as required in a square a shape as possible

#Seperate VR59 into two recording sessions, for the 17th and 18th to allow comparison


#now get units from these
units = myexp.select_units(
    group="good",
    max_depth=1500,
    name="cortex",
    uncurated=False
)


#Now that we have selected the required units, let us plot thespike rates according to LED on
hits =myexp.align_trials(
    ActionLabels.correct,
    Events.led_on,
    "spike_rate",
    duration=4,
    units=units
)

#First set the confidence interval for the graph (This significantly increases time to run!)
#ci = 95

#Now run the code to enumerate over the exp.
#The code will plot the spike rates of every unit selected above aligned to the defined event
#For every session

for s, session in enumerate(myexp):
    name = session.name

    spike_rate.per_unit_spike_rate(hits[s], ci="sd")
    plt.suptitle(
        f"Session {name} - per-unit across-trials firing rate (aligned to LED on)",
    )
    plt.set_size_inches(10, 10)
    plt.show() #Plot this information
