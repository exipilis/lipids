import os
from keras.callbacks import ModelCheckpoint
from multiprocessing import cpu_count

from model import seg_model
from sequence import SegmenterSequence

class_count = 2
steps = 300
epochs = 50
batch_size = 16

weights_filename = 'checkpoints/weights.h5'

model = seg_model(class_count)

model.summary()

try:
    model.load_weights(weights_filename)
    print('weights loaded from ' + weights_filename)
except (OSError, ValueError):
    print('weights random')

model_shape = model.input_shape[1:]

train_images = ['data/train/' + s for s in os.listdir('data/train/') if s.endswith('.jpg')]

train_generator = SegmenterSequence(model_shape, train_images, class_count, batch_size=batch_size)
# val_generator = SegmenterSequence(model_shape, val_dataset, class_count, batch_size=batch_size)

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps,
    epochs=epochs,
    callbacks=[ModelCheckpoint(weights_filename)],
    # validation_data=val_generator,
    validation_steps=steps / 10,
    max_queue_size=train_generator.batch_size * 2,
    workers=cpu_count(),
    use_multiprocessing=True
)
