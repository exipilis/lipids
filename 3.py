import numpy as np
from keras import Model
from keras.models import load_model

encoder_decoder = load_model('autoenc_m0s1_norm.h5')

encoder = Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[6].output)
encoder.compile('adam', 'mse')
encoder.summary()

data = np.load('data/meat_1fix_m0s1.npy').astype(np.float32)
data = np.reshape(data, (-1, 81))

norm = np.sqrt(np.sum(data ** 2, axis=-1)).reshape((-1, 1))
data = data / norm
min_d = np.min(data)
max_d = np.max(data)
data = 2 * (data - min_d) / (max_d - min_d) - 1

predictions = encoder.predict(data, batch_size=2 ** 14)

np.save('predictions.npy', predictions)
