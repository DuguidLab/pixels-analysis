import matplotlib.pyplot as plt

from pixtools.utils import save
from setup import fig_dir, exp



ses = 0
Fs = 30000
data = exp[ses].get_spike_data()[0]

NFFTs = [1000, 10000, 100000]
channel = 50

_, axes = plt.subplots(len(NFFTs), 1, sharex=True, sharey=True)

for i, NFFT in enumerate(NFFTs):
    axes[i].specgram(data.values[:30000 * 30, channel], NFFT=NFFT, Fs=Fs)

assert 0
save(fig_dir / "raw_specgram")
