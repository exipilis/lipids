import numpy as np
from imageio import imread, imsave
from keras.models import load_model
from scipy.misc import imresize

model = load_model('checkpoints/weights.h5')

img = imread('data/train/0cdf5b5d0ce1_01.jpg', pilmode='RGB')
img = imresize(img, (224, 224))

x = np.zeros((1, 224, 224, 3))
x[0] = img / 127.5 - 1

y = model.predict(x)

mask = y[0, :, :, 0]
mask = np.dstack([mask, mask, mask])

imsave('/tmp/result.jpg', mask * img)
