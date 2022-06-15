# This script will assess the noise in the recordings from the cortex, to allow us to decide which recordings to use
# Import required packages and data from base files
# NB: I changed how unit_depths() works in my local copy to remove the reference to rec_num



from turtle import fd

from matplotlib.pyplot import suptitle, xlabel
from base import *
from noisebase import *
from channeldepth import *
from matplotlib.cm import ScalarMappable


import numpy as np
import pandas as pd
import seaborn as sns


# Now select the experiment we want to use
# Here we are taking the data from the 17th.


# Now from this data, give us all units that are good and within the cortex
units = myexp.select_units(group="good", max_depth=1200, name="m2")

# #Now that we have the units selected, let us assess the noise
myexp.assess_noise()

# now read these dataframes in from the json files just created
# If we want to change the data read in, simply change the experimental call in noisebase!
# May also add more experiments to the list here if desired!

noise = []
for s, session in enumerate(myexp):
    for i in range(len(session.files)):
        path = session.processed / f"noise_{i}.json"
        with path.open() as fd:
            ses_noise = json.load(fd)
        date = datetime.datetime.strptime(session.name[:6], "%y%m%d")
        for SD in ses_noise["SDs"]:
            noise.append((session.name, date, name, SD))


df = pd.DataFrame(noise, columns=["session", "date", "project", "SDs", "median SD"])

# Now read in the metadata containing channel coordinates and convert it to dataframe format

depth1 = meta_spikeglx(
    myexp, 0
)  # give us data for session1, session two will be the same so perhaps we can simply clone this??
depth1 = depth1.to_dataframe()

depth = pd.concat([depth1, depth1.copy()], axis=0)

# ##Can use the code below to print the individual datapoints rather than just the median, with each point representing the noise from each channel##
# #First requires we convert the dataframe to longform (Check this once the fixed noise assessment has completed running)

df2 = df.set_index("date")["SDs"].apply(pd.Series).stack()
df2 = df2.reset_index()
df2.columns = ["date", "channel", "SDs"]
print(df2)
# #Now add the required coordinates from the metadata
# df2=pd.concat([df2, depth], axis=1).reset_index()
# print(df2)


# Channel depth is simply organised in order of the noise json, so let us simply use the channel column in df2 as a binned category
fig, ax = plt.subplots()
p = sns.stripplot(
    x="date", y="SDs", data=df2, palette="Spectral", ax=ax, hue="channel"
).set(title="Noise for Subject VR59")

ax.set_xticklabels(
    [t.get_text().split("T")[0] for t in ax.get_xticklabels()]
)  # Remove zeroes from date line
ax.legend_.remove()  # Remove the huge autogenerated legend

# Now will add a colour bar to change hue based on channel coordinates


# This code adds a legend for the colourmap
cmap = plt.get_cmap("Spectral")
norm = plt.Normalize(df2["channel"].min(), df2["channel"].max())
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.ax.set_title('"Channel Num"')


plt.show()