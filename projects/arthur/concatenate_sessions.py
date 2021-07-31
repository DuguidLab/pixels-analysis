"""
Concatenate npx recording sessions/segments, and get a large combined session.
"""

import numpy as np
import spikeextractors as se
import spikesorters as ss

from pixels import Experiment
from pixels.behaviours.reach import Reach
from spikeinterface import NumpyRecording, NumpySorting
from spikeinterface import concatenate_recordings

rec = Experiment(
    'HFR29',  # This can be a list
    Reach,
    '/home/s2120426/duguidlab/visuomotor_control/neuropixels',
    '/home/s2120426/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

self = rec[1]

for rec_num, recording in enumerate(self.files):
    output = self.processed / f'sorted_{rec_num}'
    segment1 = self.raw / f"210624_HFR29_g0_t0.imec{rec_num}.ap.bin"
    segment2 = self.raw / f"210624_HFR29_part3_g0_t0.imec{rec_num}.ap.bin"
    recording1 = se.SpikeGLXRecordingExtractor(file_path=segment1)
    recording2 = se.SpikeGLXRecordingExtractor(file_path=segment2)
    multirec = se.MultiRecordingTimeExtractor([recording1, recording2])
    ss.run_kilosort3(recording=multirec, output_folder=output)

#assert False
#con_rec = concatenate_recodings(rec[1], rec[2])
#print(con_rec) 
#s = con_rec.get_num_samples(segment_index=0)
#print(f'segment {0} num_samples {s}')
