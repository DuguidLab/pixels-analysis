import os

os.environ['KILOSORT3_PATH'] = '/opt/neuropixels/Kilosort'

import matplotlib.pyplot as plt
import seaborn as sns

from setup import exp, ActionLabels, Events

#duplicates = []
#known_sesnames = []
#for session in exp:
#    if session.name in known_sesnames:
#        duplicates.append(session)
#    else:
#        known_sesnames.append(session.name)
#for session in duplicates:
#    exp.sessions.remove(session)


#exp.get_session_by_name("211030_VR47").process_behaviour()
exp.get_session_by_name("211030_VR47").sort_spikes()
exp.get_session_by_name("211101_VR47").sort_spikes()
exp.get_session_by_name("211028_VR49").sort_spikes()

#exp.sort_spikes()
#exp.process_spikes()
#try:
#    exp.process_lfp()
#except:
#    pass
#exp.assess_noise()
#exp.extract_videos()
#exp.process_behaviour()


# Check behavioural data alignment
if False:
    behaviour = exp.align_trials(
        ActionLabels.correct_left | ActionLabels.correct_right,
        Events.led_off,
        'behavioural',
        duration=10,
    )

    channel = "/'ReachLEDs'/'0'"
    session = 2

    fig, axes = plt.subplots(10, 1, sharex=True)
    trials = range(10)

    for trial in trials:
        sns.lineplot(
            data=behaviour[session][channel][trial],
            estimator=None,
            style=None,
            ax=axes[trial]
        )

    plt.show()

# Check spike rate alignment
if False:
    units = exp.select_units(
        min_depth=None,
        uncurated=True,
        name="all_uncurated",
    )

    rates = exp.align_trials(
        ActionLabels.correct_left | ActionLabels.correct_right,
        Events.led_off,
        'spike_rate',
        duration=10,
        units=units,
    )

    session = 0
    trial = 10
    num_units = min(len(units[session]), 10)
    fig, axes = plt.subplots(num_units, 1, sharex=True)
    assert 0

    for u in range(num_units):
        unit = units[session][u + 10]
        sns.lineplot(
            data=rates[session][unit],
            estimator=None,
            style=None,
            ax=axes[u]
        )

    plt.show()
