import json

import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush
from pixels.behaviours.pushpull import PushPull
from pixels.behaviours.reach import Reach
from pixels.behaviours.no_behaviour import NoBehaviour

import sys
sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")
from pixtools import utils
from pixtools import clusters

from base import *


noise_tests = Experiment(
    ["noisetest1"],
    NoBehaviour,
    "~/duguidlab/visuomotor_control/neuropixels",
)

noise_test_unchanged = Experiment(
    ["noisetest_unchanged"],
    NoBehaviour,
    "~/duguidlab/visuomotor_control/neuropixels",
)

noise_test_nopi = Experiment(
    ["noisetest_nopi"],
    NoBehaviour,
    "~/duguidlab/visuomotor_control/neuropixels",
)

noise_test_no_caps = Experiment(
    "VR50",
    Reach,
    "~/duguidlab/visuomotor_control/neuropixels",
)

#This contains the list of experiments we want to plot noise for, here only interested in the reaching task
#i.e., VR50
#Remember to change this in base.py
exps = {
    "reaching": myexp
}