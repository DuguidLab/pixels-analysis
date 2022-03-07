#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:10:30 2022

@author: Aidan (s1735718)
"""
#Import experimental instance
from base import *

# This aligns, crops and downsamples behavioural data.
#myexp.process_behaviour()

## This aligns, crops and downsamples LFP data.
#myexp.process_lfp()
#
## This aligns, crops, and downsamples spike data.
#myexp.process_spikes()
#
## This runs the spike sorting algorithm and outputs the results in a form usable by phy.
myexp.sort_spikes()

## This extracts posture coordinates from TDMS videos using DeepLabCut
#config = '/path/to/this/behaviours/deeplabcut/config.yaml'
#myexp.process_motion_tracking(config)
## If you also want to output labelled videos, pass this keyword arg:
#myexp.process_motion_tracking(config, create_labelled_video=True):
## This method will convert the videos from TDMS to AVI before running them through DLC.
## If you just want the AVI videos without the DLC, you can do so directly:
#myexp.extract_videos()