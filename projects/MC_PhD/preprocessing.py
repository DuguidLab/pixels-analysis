import os

os.environ['KILOSORT3_PATH'] = '/opt/neuropixels/Kilosort'

from setup import exp

duplicates = []
known_sesnames = []
for session in exp:
    if session.name in known_sesnames:
        duplicates.append(session)
    else:
        known_sesnames.append(session.name)
for session in duplicates:
    exp.sessions.remove(session)


exp.sort_spikes()
exp.process_spikes()
#try:
#    exp.process_lfp()
#except:
#    pass
exp.assess_noise()
#exp.extract_videos()
exp.process_behaviour()
