import os
import random

import numpy as np
import scipy.ndimage as si
import tensorflow as tf

from lfw.affinegenerator import random_affine as ra


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


root_dir = ""  # You're directory here

file_list = []

for subdir, dir, files in os.walk(root_dir):
    for file in files:
        file_list.append(os.path.join(subdir, file))

random.shuffle(file_list)

print(len(file_list))

mid_point = int(len(file_list)/2)

train_list = file_list[:mid_point]
test_list = file_list[mid_point:]


writer = tf.python_io.TFRecordWriter("") # You're record file name here

for path in file_list:
    transform = np.ndarray.flatten(ra.random_affine())[0:8]
    img = si.imread(path)
    height = img.shape[0]
    width = img.shape[1]
    img_raw = img.tostring()
    example = tf.train.Example(
        features = tf.train.Features(
            feature={
                'image': _bytes_feature(img_raw),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'transform': _float_feature(transform)
            }
        )
    )
    writer.write(example.SerializeToString())
writer.close()

