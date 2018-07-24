import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense

data = np.load('data/meat_1fix_original.npy').astype(np.float32)
data = np.reshape(data, (-1, 81))
norm = np.sqrt(np.sum(data**2, axis=-1)).reshape((-1, 1))
data = data / norm
min_d = np.min(data)
max_d = np.max(data)
print(min_d, max_d)
data = 2 * (data - min_d) / (max_d - min_d) - 1  # -1..1


# reshaped_y = np.reshape(data, (-1, 81))


def autoenc() -> Model:
    inp = Input((81,))

    x = Dense(32, activation='sigmoid')(inp)
    x = Dense(32, activation='sigmoid')(x)
    x = Dense(32, activation='sigmoid')(x)
    x = Dense(32, activation='sigmoid')(x)
    x = Dense(32, activation='sigmoid')(x)
    x = Dense(2, activation='sigmoid')(x)
    x = Dense(32, activation='sigmoid')(x)
    x = Dense(32, activation='sigmoid')(x)
    x = Dense(32, activation='sigmoid')(x)
    x = Dense(32, activation='sigmoid')(x)
    x = Dense(81, activation='tanh')(x)

    m = Model(inputs=inp, outputs=[x, x])
    m.compile('adam', ['mae', 'mse'], loss_weights=[1, 1])

    return m


model = autoenc()
model.summary()

try:
    model.load_weights('autoenc.h5')
    print('load weights')
except (IOError, ValueError):
    print('random weights')

# data = np.random.standard_exponential(data.shape)
# np.save('data/random.npy', data)

model.fit(x=data, y=[data, data], batch_size=512, epochs=500, callbacks=[ModelCheckpoint('autoenc.h5')])
