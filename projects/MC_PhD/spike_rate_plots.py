from pixtools import spike_rate

from setup import *

sns.set(font_scale=0.4)
duration = 4

units = exp.select_units(
    max_depth=1500,
    name="up_to_1500",
)

correct = exp.align_trials(
    ActionLabels.correct_left | ActionLabels.correct_right,
    Events.led_off,
    'spike_rate',
    duration=duration,
    units=units,
)

#ci = "sd"
ci = 95

for s, session in enumerate(exp):
    name = session.name

    # per unit
    spike_rate.per_unit_spike_rate(correct[s], ci=ci)
    plt.suptitle(f'Session {name} - per-unit across-trials firing rate (aligned to grasp)')
    utils.save(fig_dir / f'unit_spike_rate_correct_{duration}s_{name}_sd')

    ## per trial
    #spike_rate.per_trial_spike_rate(correct[s], ci=ci)
    #plt.suptitle(f'Session {name} - per-trial across-units firing rate (aligned to grasp)')
    #utils.save(fig_dir / f'trial_spike_rate_correct_{duration}s_{name}_sd')
