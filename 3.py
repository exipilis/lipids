import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras import Model
from keras.models import load_model

encoder_decoder = load_model('autoenc.h5')

encoder = Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[6].output)
encoder.compile('adam', 'mse')
encoder.summary()

data = np.load('data/meat_1fix_der.npy').astype(np.float16)
data = np.reshape(data, (-1, 81))
min_d = np.min(data)
max_d = np.max(data)
data = 2 * (data - min_d) / (max_d - min_d) - 1

predictions = encoder.predict(data, batch_size=9192)

sns.set(context="paper", style="white")
fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(data[:, 0], data[:, 1], cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("UMAP", fontsize=18)

plt.show()
