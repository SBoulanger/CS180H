import pandas as pd
import numpy as np
import tensorflow as tf
import math
import os
from PIL import Image

IMG_SIZE = 200  # 200x200 pixels for each image
DATA_PATH = '../../Datasets/four_shapes'
SHAPES = ('circle', 'square', 'star', 'triangle')

# first load in the images via pandas
mypath = os.path.join(DATA_PATH, SHAPES[0])


def _get_shape_files(shape):
    shape_files = []
    path = os.path.join(DATA_PATH, shape)
    _, _, files = next(os.walk(path), (None, None, []))
    shape_files.extend(files)
    return shape_files


def get_shape_data(shape):
    files = _get_shape_files(shape)
    data_as_list = [np.asarray(Image.open(os.path.join(DATA_PATH, shape, f)).convert('L')) for f in files]
    np_data = np.asarray(data_as_list)
    tf_dataset = tf.data.Dataset.from_tensor_slices(np_data)
    return tf_dataset


def main():
    slices = get_shape_data('circle')
    next_item = slices.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(next_item))
            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
    main()