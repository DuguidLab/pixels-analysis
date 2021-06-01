from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import tensorflow as tf
from pathlib import Path

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events


mice = [
    'C57_724',
    'C57_1288723',
    'C57_1288727',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures')

def save(name):
    plt.gcf().savefig(fig_dir / name, bbox_inches='tight', dpi=300)


## run settings
ses = exp[0]
rec_num = 0


## define model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


## get x data
cache = ses.interim / 'cache' / 'firing_rates.npy'
if cache.exists():
    x = np.load(cache)
else:
    times = ses._get_spike_times()[rec_num]
    times = times / 30000
    duration = round(times.max().max())
    x_eval = np.arange(0, duration, 0.001)
    x = np.zeros((len(times.columns), duration * 1000))

    for i, unit in enumerate(times.columns):
        kde = scipy.stats.gaussian_kde(times[unit].dropna(), bw_method=0.01)
        rate = kde(x_eval) * len(times[unit])
        x[i, :] = rate / rate.max()  # normalise

    np.save(cache, x)


## get y data
raise Exception


## train
model.fit(x, y, epochs=3)


## test
x_test, y_test = load_data('train')
x_test = x_test / 255
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

# the model can be saved:
#model.save('mnist.model')
#new_model = tf.keras.models.load_model('mnist.model')

# make predictions
predictions = model.predict(x_test)
_, axes = plt.subplots(1, 10)
for i in range(10):
    axes[i].imshow(x_test[i].reshape((28, 28)))
    axes[i].set_title(np.argmax(predictions[i]))
plt.show()
