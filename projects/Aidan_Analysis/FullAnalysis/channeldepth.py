#This file will read in channel depth information from the available metadata
#First import required packages
 
from base import *
import probeinterface as pi

from reach import Session

#Find the metadata containing the depth of the channels used in the recording
#Remember must index into experiment and then files to find the spike metadata
#Shall create a function to do this
def meta_spikeglx(exp, session):
    meta=exp[session].files[0]["spike_meta"]
    data_path= myexp[session].find_file(meta)
    data=pi.read_spikeglx(data_path)

    return data





