import numpy as np
import tensorflow as tf

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100

TEST_LENGTH = 19962
TRAIN_LENGTH = 162770

TRAIN_DIR = "celebA-train.tfrecords"
TEST_DIR = "celebA-eval.tfrecords"


# Reads and parses examples from a tf record dataset with the format specified by the parse module
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'transform': tf.FixedLenFeature([8], tf.float32)
        })

    image = tf.decode_raw(features['image'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    transform = tf.cast(features['transform'], tf.float32)
    
    image_shape = tf.stack([height, width, 3], 0)

    image_reshape = tf.reshape(image, image_shape)

    image_reshape_final = tf.image.resize_image_with_crop_or_pad(image_reshape, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
    
    image_final = tf.cast(image_reshape_final, tf.float32)

    return image_reshape_final, transform

# Produces an image batch for the CNN
def _generate_image_batch(image, transform, min_queue_examples, batch_size, shuffle):

    num_prefetch_threads = 2

    if shuffle:
        img_batch, transform_batch = tf.train.shuffle_batch(
            [image, transform],
            batch_size=batch_size,
            num_threads = num_prefetch_threads,
            capacity=min_queue_examples+ 3*batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        img_batch, transform_batch = tf.train.batch(
            [image, transform],
            batch_size=batch_size,
            num_threads=num_prefetch_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return img_batch, transform_batch

# Pack images into batches without shuffling for evaluation purposes
def normal_inputs(eval_data, batch_size):

    if not eval_data:
        record_file = TRAIN_DIR
        num_examples = TRAIN_LENGTH
    else:
        record_file = TEST_DIR
        num_examples = TEST_LENGTH

    filename_queue = tf.train.string_input_producer([record_file])

    image, transform = read_and_decode(filename_queue)

    min_fraction_of_examples_in_queue = 0.4  # Use to be 0.4
    min_queue_examples = int(num_examples * min_fraction_of_examples_in_queue)

    return _generate_image_batch(image, transform, min_queue_examples, batch_size, shuffle=False)

# Same as normal_inputs() but randomises batches for training
def standardized_input(batch_size):

    filename_queue = tf.train.string_input_producer([TRAIN_DIR])
    image, transform = read_and_decode(filename_queue)
    

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(TRAIN_LENGTH * min_fraction_of_examples_in_queue)

    return _generate_image_batch(image, transform, min_queue_examples, batch_size, shuffle=True)


