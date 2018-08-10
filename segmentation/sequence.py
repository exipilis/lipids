import random

import numpy as np
from imageio import imread, imsave
from keras.utils import Sequence
from scipy.misc import imresize


class SegmenterSequence(Sequence):
    def __init__(self,
                 image_shape: tuple,
                 image_filenames: list,
                 classes_count: int,
                 batch_size: int = 32,
                 shuffle: bool = True):

        self.batch_size = batch_size
        self.image_filenames = image_filenames
        self.classes_count = classes_count

        self.image_shape = image_shape
        self.mask_shape = image_shape[0:2] + (self.classes_count,)

        self.shuffle = shuffle
        self.batches_fetched = 0
        self.debug = False

        self.len_images = len(self.image_filenames)

    def __len__(self):
        return np.math.ceil(self.len_images / self.batch_size)

    def __getitem__(self, index):
        s = index * self.batch_size
        e = s + self.batch_size
        self.batches_fetched += 1

        if not (self.batches_fetched % len(self)) and self.shuffle:
            random.shuffle(self.image_filenames)

        filenames_batch = [self.image_filenames[i % self.len_images] for i in range(s, e)]

        x_batch = np.zeros((self.batch_size,) + self.image_shape, dtype=np.float32)
        y_batch = np.zeros((self.batch_size,) + self.mask_shape, dtype=np.uint8)

        for i, image_fn in enumerate(filenames_batch):
            img = imread(image_fn, pilmode='RGB')
            # final resize
            img = imresize(img, self.image_shape)
            if self.debug:
                imsave('/tmp/%s.jpg' % i, img)

            mask_fn = image_fn.replace('/train/', '/train_masks/').replace('.jpg', '_mask.gif')
            mask_img = imread(mask_fn)
            mask_img = imresize(mask_img, self.image_shape)
            mask_img = np.round(mask_img / 255).astype(np.uint8)

            mask = np.zeros(self.mask_shape, dtype=np.uint8)
            mask[:, :, 0] = mask_img
            mask[:, :, 1] = 1 - mask_img

            if self.debug:
                imsave('/tmp/%s_mask.gif' % i, mask * 255)

            x_batch[i] = img
            y_batch[i] = mask

        x_batch = x_batch / 127.5 - 1

        return x_batch, y_batch
