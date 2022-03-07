import sys
sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis") #Adds the location of the pixtools folder to the path

from pixtools import spike_rate

from base import *

sns.set(font_scale=0.4)
duration = 4

myexp.set_cache("overwrite")

# This selects the specific neuronal units will be included in the plots
units = myexp.select_units(
    group="good",  # Gives quality of units, here good only
    max_depth=1500,  # Give us units above a depth of 1500um
    name="cortex",  # The name to cache the info under
#If dealing with pre-phy data be sure to specify "uncurated = TRUE" in the select_units function
)

correct = myexp.align_trials(
    ActionLabels.correct,
    Events.led_off,
    "spike_rate",
    duration=duration,
    units=units,
)

ci = "sd"
#ci = 95

for s, session in enumerate(myexp):
    name = session.name

    # per unit
    spike_rate.per_unit_spike_rate(correct[s], ci=ci)
    plt.suptitle(
        f"Session {name} - per-unit across-trials firing rate (aligned to grasp)"
    )
    utils.save(fig_dir / f"unit_spike_rate_correct_{duration}s_{name}_{ci}")

    ## per trial
    # spike_rate.per_trial_spike_rate(correct[s], ci=ci)
    # plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to grasp)')
    # utils.save(fig_dir / f'trial_spike_rate_correct_{duration}s_{name}_sd')
