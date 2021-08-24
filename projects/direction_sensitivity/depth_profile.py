import matplotlib.pyplot as plt
import seaborn as sns

from pixels.behaviours.pushpull import PushPull
from pixtools import clusters, utils

from setup import exp, rec_num, units, fig_dir

sns.set(font_scale=0.4)
fig_dir = fig_dir / 'DS'

# all clusters
fig = clusters.depth_profile(exp, curated=False)
plt.ylim([2000, - 250])
plt.suptitle('Cluster depth profile uncurated')
utils.save(fig_dir / f'cluster_depth_profile_uncurated')

#fig = clusters.depth_profile(exp, curated=True)
#plt.ylim([-250, 2000])
#plt.suptitle('Cluster depth profile curated')
#utils.save(fig_dir / f'cluster_depth_profile_curated')

# post-curation good units in the brain
#fig = clusters.depth_profile(exp, curated=True, group='good')
#plt.suptitle('Cluster depth profile good units in brain')
#plt.ylim([2000, 0])
#utils.save(fig_dir / f'cluster_depth_profile_neurons')
