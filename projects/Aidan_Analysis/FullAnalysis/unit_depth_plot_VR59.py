#Import data from base 

from base import *

#Import required packages
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")
from pixtools import clusters #A series of functions that will allow plotting of depth profiles

#Select which units from the data we would like to plot
#To get an overview of the data, let us select all units of good quality and plot their depth
#Select only from recording from the 17th
session1 = myexp.get_session_by_name("220217_VR59") #It appears this is the line that is screwing with things, why does it make an unindexable item??

units = session1.select_units(
    group="good",
    uncurated = False,
)

#Now plot a graph of increasing depth vs unit 
#First select the depth profile to analyse, function will then automatically plot the required graph as a scatter
#May need to alter this to remove use of rec_num!# Or perhaps simply set rec_num to 0?

rec_num = 0

depth = clusters.depth_profile(
    myexp,
    group="good",
    curated = True
)


plt.show()