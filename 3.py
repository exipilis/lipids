import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras import Model
from keras.models import load_model
from sklearn.cluster import KMeans

encoder_decoder = load_model('autoenc.h5')

encoder = Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[6].output)
encoder.compile('adam', 'mse')
encoder.summary()

data_orig = np.load('data/meat_1fix_original.npy').reshape((-1, 81))
colors = np.sqrt(np.sum(data_orig ** 2, axis=-1))

data = np.load('data/meat_1fix_m0s1.npy').astype(np.float32)
data = np.reshape(data, (-1, 81))

norm = np.sqrt(np.sum(data**2, axis=-1)).reshape((-1, 1))
data = data / norm
min_d = np.min(data)
max_d = np.max(data)
data = 2 * (data - min_d) / (max_d - min_d) - 1

predictions = encoder.predict(data, batch_size=2**14)

print(predictions.shape)

kmeans = KMeans(n_clusters=2, n_init=50, max_iter=300, tol=1e-5, n_jobs=-1).fit(predictions)
print(kmeans.cluster_centers_)

sns.set(context="paper", style="white")
fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(predictions[:, 0], predictions[:, 1], c=colors, cmap="viridis", s=0.1)
plt.colorbar()
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^', c='red')
plt.title("Autoencoder", fontsize=18)

plt.show()
