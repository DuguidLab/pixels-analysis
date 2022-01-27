from pixtools import clusters

from setup import *

sns.set(font_scale=0.4)

# post-curation good units in the brain
clusters.depth_profile(exp, group='good')
plt.suptitle('Depth profile of good units')
plt.ylim([3000, 0])
utils.save(fig_dir / f'depth_profile_good_units', nosize=True)
