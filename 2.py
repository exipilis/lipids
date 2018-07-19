import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense

data = np.load('data/meat_1fix_der.npy').astype(np.float16)
data = np.reshape(data, (-1, 81))
min_d = np.min(data)
max_d = np.max(data)
print(min_d, max_d)
data = 2 * (data - min_d) / (max_d - min_d) - 1


def autoenc() -> Model:
    inp = Input((81,))

    x = Dense(64, activation='sigmoid')(inp)
    x = Dense(64, activation='sigmoid')(x)
    x = Dense(64, activation='sigmoid')(x)
    x = Dense(64, activation='sigmoid')(x)
    x = Dense(64, activation='sigmoid')(x)
    x = Dense(2, activation='sigmoid')(x)
    x = Dense(64, activation='sigmoid')(x)
    x = Dense(64, activation='sigmoid')(x)
    x = Dense(64, activation='sigmoid')(x)
    x = Dense(64, activation='sigmoid')(x)
    x = Dense(81, activation='tanh')(x)

    m = Model(inputs=inp, outputs=x)
    m.compile('adam', 'mse')

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

model.fit(x=data, y=data, batch_size=512, epochs=100, callbacks=[ModelCheckpoint('autoenc.h5')])
