import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

data_orig = np.load('data/meat_1fix_original.npy').reshape((-1, 81))
colors = np.sqrt(np.sum(data_orig ** 2, axis=-1))

predictions = np.load('predictions.npy')
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
