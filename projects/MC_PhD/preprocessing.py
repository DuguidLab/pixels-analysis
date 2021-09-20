import os

os.environ['KILOSORT3_PATH'] = '/opt/neuropixels/Kilosort'

from setup import exp

#exp.sort_spikes()
#try:
#    exp.process_spikes()
#except:
#    pass
#try:
#    exp.process_lfp()
#except:
#    pass
#exp.assess_noise()
exp.extract_videos()
exp.process_behaviour()
