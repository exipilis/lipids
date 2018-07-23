import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from imageio import imread


def read_files(data_dir):
    files = [s for s in os.listdir(data_dir) if s.endswith('.jpg')]

    files.sort(key=lambda s: int(s.split('.')[0].split('_')[-1]))

    images = [np.array(imread(data_dir + fn, pilmode='L'), dtype=np.float32) for fn in files]
    reshaped = np.moveaxis(images, 0, -1)

    print(reshaped.shape)

    np.save('data/meat_1fix_original.npy', reshaped)

    means = [np.mean(img) for img in images]
    stds = [np.std(img) for img in images]

    images_norm = [(img - means[i]) / stds[i] for i, img in enumerate(images)]

    images_norm = np.array(images_norm)

    reshaped = np.moveaxis(images_norm, 0, -1)

    np.save('data/meat_1fix_m0s1.npy', reshaped)


def save_tsv():
    data = np.load('data/meat_1fix.npy')
    reshaped = np.reshape(data, (-1, 81))
    print(reshaped.shape)
    np.savetxt('data/meat_1fix.tsv', reshaped, delimiter='\t', fmt='%d')


def make_umap():
    sns.set(context="paper", style="white")

    data = np.load('data/meat_1fix.npy').astype(np.float32)
    data = np.reshape(data, (-1, 81))
    min_d = np.min(data)
    max_d = np.max(data)
    data = 2 * (data - min_d) / (max_d - min_d) - 1

    colors = np.sum((data / 2 + 0.5) ** 2, axis=-1)

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(data)

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap="Spectral", s=0.1)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("UMAP", fontsize=18)

    plt.show()


def der():
    data = np.load('data/meat_1fix.npy').astype(np.float16)
    d = np.gradient(data, axis=-1)

    np.save('data/meat_1fix_der.npy', d)
    np.savetxt('data/meat_1fix_der.tsv', np.reshape(d, (-1, 81)), delimiter='\t', fmt='%0.1f')


if __name__ == '__main__':
    read_files('data/Meat_1fix/')
    # save_tsv()
    # make_umap()
    # der()
