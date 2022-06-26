#Import our experimental data from base

from base import *

#The first time this is run it will take a while!! There is a lot of data to go through and many steps##
#It will pull from raw data and output to processed data.

#First we will select only the recording from the 17th and assign this to a variable


#First run Kilosort on data to prepare for phy curation
#Remember to open phy after to generate the cluster info file!!
myexp.sort_spikes()

# #Then process behavioural data 
myexp.process_behaviour()

# # #Then alicn, crop, and downsample spike data
myexp.process_spikes()

# # #Now do the same with local field potential data
myexp.process_lfp()

#Opetionally analyse the noise of the recordings
myexp.assess_noise()