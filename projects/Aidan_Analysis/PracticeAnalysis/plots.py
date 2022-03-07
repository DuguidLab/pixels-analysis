#Import all required packages
from base import *
from rasterplot import *

#myexp.set_cache("overwrite") #Overwrite the cache to give a clean slate to work with

#Now set up our data for plotting, first by specifying which units shall be included in the plots - here all good ones from VR46!
units = myexp.select_units(
    group="good",  # Gives quality of units, here good only
    max_depth=1500,  # Give us units above a depth of 1500um
    name="cortex",  # The name to cache the info under 
)

#Now from this unit selection, give us all data where the mouse correctly reached for the probe when the led turned on
# data = myexp.align_trials(
#     ActionLabels.correct,  # This selects which trials we want
#     Events.led_on,  # This selects what event we want them aligned to
#     "spike_times",  # And this selects what kind of data we want, need the time the spikes occured to plot a raster graph
#     duration=4,  # Gives us two seconds on either side of the event
#     units=units,
# )

assert units[0]

#Now our data is prepared we can plot raster by trial, ensure you index into data beforehand!!
#Using the subplots class, we can overlay different data in the same graph (in diff. colours)
#Start is the initial offset

#per_trial_raster(data[0], sample=None, start=0, subplots=None, label=True)

#Can also plot raster data by Unit, where each plot is a unit
#per_unit_raster(data[0], sample=None, start=0, subplots=None, label=True)

###Now say we wanted to split data by left and right reaches and plot both of these on one graph
#This can be done with the subplots function, and by changing what units are selected under action labels.

#First select all data that is correct from the right, and left sides

dataR = myexp.align_trials(
    ActionLabels.correct_right,  # This selects which trials we want
    Events.led_on,  # This selects what event we want them aligned to
    "spike_times",  # And this selects what kind of data we want, need the time the spikes occured to plot a raster graph
    duration=4,  # Gives us two seconds on either side of the event
    units=units,
)

dataL = myexp.align_trials(
    ActionLabels.correct_left,  # This selects which trials we want
    Events.led_on,  # This selects what event we want them aligned to
    "spike_times",  # And this selects what kind of data we want, need the time the spikes occured to plot a raster graph
    duration=4,  # Gives us two seconds on either side of the event
    units=units,
)

#Now set the data we wish to subplot as a variable, then use this as the argument for the Subplot function
subplot = per_unit_raster(dataL[0], sample=None, start=0, subplots=None, label=True)
per_unit_raster(dataR[0], sample=None, start=0, subplots=subplot, label=True)
plt.show() #Display the plot rather than saving it. 