import tensorflow as tf
import math
from operator import mul
import functools
from TPS import FINAL_TPS  as tps

import lfw_input

# Learning parameters
NUM_EPOCHS_PER_DECAY = 5
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.9999

# Number of landmarks to find
NO_LANDMARKS = 10

# Gamma ensures landmarks are adequatly separated - the higher the gamma the greater the landmark diversity 
GAMMA_DEFAULT = 100

# Amount to crop off images after performing transformations
CROP_SIZE = 20

# Amount to downsample images in each direction
PATCH_SIZE = 5

# Parameters for the TPS transforms used before CNN

# Affine parameters
SCALE_1 = 0
SCALE_2 = 0.05
ROT_1 = 0
ROT_2 = 0.34
TFORM_1 = 0
TFORM_2 = 0.1

# Control point parameters
SIGMA_G1 = 0.001
SIGMA_G2 = 0.005
SIGMA2_G1 = 0.001
SIGMA2_G2 = 0.01

# Number of control points in each direction for TPS
CX = 5
CY = 5

FLAGS = tf.app.flags.FLAGS

# Batch size
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")

# Number of examples for each dataset
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = lfw_input.TRAIN_LENGTH
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = lfw_input.TEST_LENGTH

# Calculates the control points for a TPS transformation, note coordinates are relative to a fixed grid
def choose_control_points(nx, ny, batch_size, std, std2):

    rand_variation = tf.random_normal(shape=tf.stack([batch_size, nx, ny,2]), stddev=std)
    peturb_chance = tf.random_uniform(shape=tf.stack([batch_size, nx, ny, 2]))
    perturbs = tf.random_normal(shape=tf.stack([batch_size, nx, ny, 2]), stddev=std2)
    perturbed_variation = rand_variation + perturbs
    bool_table = tf.less_equal(peturb_chance, 0.5)
    
    coords = tf.where(bool_table, x=perturbed_variation, y=rand_variation)
    coords = tf.reshape(coords, (batch_size, -1,2))
    
    return coords

# Generates the affine componenets of TPS transforms
def choose_affine_components(batch_size, rot, sc, t):
    rot = tf.random_normal(shape=[batch_size], stddev=rot)
    x_trans = tf.random_normal(shape=[batch_size], stddev=t)
    y_trans = tf.random_normal(shape=[batch_size], stddev=t)
    x_scale = tf.random_normal(shape=[batch_size], mean=1, stddev=sc)
    y_scale = tf.random_normal(shape=[batch_size], mean=1, stddev=sc)

    c_rot = tf.cos(rot)
    s_rot = tf.sin(rot)

    affine = tf.stack([c_rot*x_scale, s_rot, x_trans, -1*s_rot, c_rot*y_scale, y_trans, tf.zeros(shape=(batch_size)), tf.zeros(shape=(batch_size)), tf.ones(shape=(batch_size))], 1)
    return tf.reshape(affine, [batch_size, 3, 3]) 

# Pads images with Gaussian values based on the mean and std of all pixel values in the image batch
# This helps to alleviate landmarks found in corners BUT means that the exact landmarks found when
# passing the same image through the CNN will change run to run - hence I am not using it at the moment.
def perform_gaussian_pad(x_img, width, height):
    shape= tf.shape(x_img)
    new_width = shape[2] + 2*width
    new_height = shape[1] + 2*height

    mean, var = tf.nn.moments(x_img, axes=[0, 1, 2, 3])
    std = tf.sqrt(var)
    
    left = tf.random_normal([shape[0],shape[1], width, shape[3]], mean=mean, stddev=std)
    right =  tf.random_normal([shape[0], shape[1], width, shape[3]], mean=mean, stddev=std)

    x_w = tf.concat([left, x_img, right], axis=2) 

    top = tf.random_normal([shape[0], height, new_width, shape[3]], mean=mean, stddev=std)
    bottom = tf.random_normal([shape[0], height, new_width, shape[3]], mean=mean, stddev=std)

    x_h = tf.concat([top, x_w, bottom], axis=1) 
 
    return x_h


# Define a convolutional layer
def conv2d(x, W, v_stride=1, h_stride=1):
    return tf.nn.conv2d(x, W, strides=[1, v_stride, h_stride, 1], padding="VALID")


# Define a pooling layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Performs softmax on 2D matrices
def spatial_softmax(x):
    shape = tf.shape(x)
    x_reshape = tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [shape[0] * shape[3], shape[1] * shape[2]])
    softmax = tf.nn.softmax(x_reshape)
    softmax = tf.transpose(tf.reshape(softmax, [shape[0], shape[3], shape[1], shape[2]]), [0, 2, 3, 1])
    return softmax

# Note trying in built xavier initializer
def conv_layer(x, dims, name, mode):
    with tf.variable_scope(name):
        W = tf.get_variable("W", dims, initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(0.0005))
        b = tf.get_variable("b", dims[-1], initializer=tf.zeros_initializer())

        ws_conv = conv2d(x, W) + b
        norm_conv= tf.layers.batch_normalization(ws_conv,training=mode, axis=3, name="batch")
        h_conv = tf.nn.relu(norm_conv)
        return h_conv

# Defines the overall structure of the CNN
def cnn(x, mode):
    x_pad = tf.pad(x, [[0,0],[2,2], [2,2],[0,0]], mode='CONSTANT') 
    c1 = conv_layer(x_pad, [5, 5, 3, 20], "conv1", mode) # Channel depth here
    c1_pool = max_pool_2x2(c1)
    c1_pad = tf.pad(c1_pool, [[0,0],[2,2],[2,2],[0,0]], mode='CONSTANT')
    c2 = conv_layer(c1_pad, [5, 5, 20, 48], "conv2", mode)
    c2_pad = tf.pad(c2, [[0,0], [1,1], [1,1], [0,0]], mode='CONSTANT')
    c2_pad = perform_gaussian_pad(c2, 1,1)
    c3 = conv_layer(c2_pad, [3, 3, 48, 64], "conv3", mode)
    c3_pad = tf.pad(c3, [[0,0], [1,1], [1,1], [0,0]], mode='CONSTANT')
    c4 = conv_layer(c3_pad, [3, 3, 64, 80], "conv4", mode)
    c4_pad = tf.pad(c4, [[0,0], [1,1], [1,1],[0,0]], mode='CONSTANT') 
    c5 = conv_layer(c4_pad, [3, 3, 80, 256], "conv5", mode)

    with tf.variable_scope("c6"):
        W = tf.get_variable("W", [3, 3, 256, NO_LANDMARKS], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [NO_LANDMARKS], initializer=tf.zeros_initializer())
        
        c5_pad = tf.pad(c5, [[0,0], [1,1], [1,1], [0,0]], mode='CONSTANT')
        ws_conv = conv2d(c5_pad, W) + b
        h_conv = spatial_softmax(ws_conv)
    return h_conv

# Gets (poetentially normalised) coordinates as a meshgrid for computing norms of image coordinates
def get_coords(height, width, norm=True):
    
    if  not norm:
        h_range = tf.range(height)
        w_range = tf.range(width)
    else:
        h_range = tf.linspace(-1.0, 1.0, height)
        w_range = tf.linspace(-1.0, 1.0, width)

    h_range = tf.expand_dims(h_range, axis=1)
    w_range = tf.expand_dims(w_range, axis=0)

    h_range = tf.cast(tf.tile(h_range, [1, width]), tf.float32)
    w_range = tf.cast(tf.tile(w_range, [height, 1]), tf.float32)

    return h_range, w_range

# Alignment loss - Punishes the CNN for finding landmarks in the transformed images that are in different places
def l_align(o_output, transforms, t_output, fp):
    
    shape = tf.shape(o_output)
    hr, wr = get_coords(shape[1], shape[2], norm=True)
    batch_size = tf.slice(shape, [0], [1])
    
    height_f = tf.cast(shape[1], tf.float32)
    width_f = tf.cast(shape[2], tf.float32)
   
    # Remember this is flat and not stacked!
    sqs = tf.expand_dims(tf.square(hr) + tf.square(wr), axis=2)
   
    # Don't need to do this anymore as we have x_s and y_s
    o_h = tf.shape(o_output)[1]
    o_w = tf.shape(o_output)[2]
    x_h, y_h = tps._transform(transforms, fp, o_output, (o_h, o_w))   
    x_f = (x_h + 1.0)*(width_f) / 2.0
    y_f = (y_h + 1.0)*(height_f) / 2.0
    x_r = tf.reshape(x_h, [FLAGS.batch_size, shape[1], shape[2], 1]) 
    y_r = tf.reshape(y_h, [FLAGS.batch_size, shape[1], shape[2], 1])
    
    # This is stacked in the batch dimension!
    sqs_t = tf.square(y_r) + tf.square(x_r)

    p1 = sqs * o_output
    p1 = tf.reduce_sum(p1)

    p2 = sqs_t * t_output
    p2 = tf.reduce_sum(p2)

    p3_h = tf.expand_dims(hr, axis=2) * o_output
    p3_h = tf.reduce_sum(p3_h, axis=[1, 2])

    p3_w = tf.expand_dims(wr, axis=2) * o_output
    p3_w = tf.reduce_sum(p3_w, axis=[1,2])

    p3_o = tf.stack([p3_h, p3_w], axis=2)
    p3_o = tf.expand_dims(p3_o, axis=2)

    p4_h = x_r * t_output
    p4_h = tf.reduce_sum(p4_h, axis=[1, 2])

    p4_w = y_r * t_output
    p4_w = tf.reduce_sum(p4_w, axis=[1, 2])

    p4_t = tf.stack([p4_h, p4_w], axis=2)
    p4_t = tf.expand_dims(p4_t, axis=3)
        
    p3_p4 = -2 * tf.matmul(p3_o, p4_t)

    p3_p4 = tf.reduce_sum(p3_p4)

    bs = tf.cast(tf.shape(o_output)[0], tf.float32)

    return (p1 + p2 + p3_p4)/(bs * tf.cast(shape[3], tf.float32))

# Diversity componenet of loss function - punishes overlapping landmarking
def l_div(output):
    filters = tf.cast(tf.shape(output)[3], tf.float32)

    # Downsample with average pool with multiplication by kernel size
    output_pool = tf.nn.avg_pool(output, ksize=[1, PATCH_SIZE, PATCH_SIZE, 1], strides=[1, PATCH_SIZE, PATCH_SIZE, 1], padding="VALID") * PATCH_SIZE * PATCH_SIZE
    output_max = tf.reduce_max(output_pool, axis=[3], keep_dims=True)
    output_sum = tf.reduce_sum(output_max, axis=[1, 2], keep_dims=True)
    aligns = filters - output_sum
    aligns_all = tf.reduce_sum(aligns)

    bs = tf.cast(tf.shape(output)[0], tf.float32)

    return aligns_all / bs


# Finds the final landmarks from the output of the CNN by taking the maximum position of each probability map
def find_indices(out):

    def find_max(x, h, w, c):
        x_t = tf.transpose(x, [2, 0, 1])
        x_flat = tf.reshape(x_t, [c, h*w])
        x_max = tf.argmax(x_flat, axis=1)
        x_ind = tf.floordiv(x_max, tf.cast(w, tf.int64))
        y_ind = x_max - x_ind*tf.cast(w, tf.int64)
        coord = tf.stack([x_ind, y_ind], axis=1)
        return coord * 2 + CROP_SIZE

    shape = tf.shape(out)

    result = tf.scan(lambda a, x: find_max(x, shape[1], shape[2], shape[3]), out, tf.zeros([shape[3], 2], dtype=tf.int64))

    return result

# Finds the final landmarks from the output of the CNN by taking the expected position of each probability map
def find_expected_indices(output):

    shape = tf.shape(output)
    hr, wr = get_coords(shape[1], shape[2], norm=False)
    
    hr_dim = tf.expand_dims(hr, 2)
    wr_dim = tf.expand_dims(wr, 2)

    output_hr = output * hr_dim
    output_wr = output * wr_dim

    output_he = tf.reduce_sum(output_hr, axis=[1,2])
    output_we = tf.reduce_sum(output_wr, axis=[1,2])
    
    coords = tf.stack([output_he, output_we], axis=2) * 2 + CROP_SIZE

    return coords


# Performs image transformations on an image batch
def perform_transform(image_batch, transform_batch):
    return tf.contrib.image.transform(image_batch, transform_batch)

# Crops all images in batch to a specified size
def batch_crop(image_batch, t_height, t_width):
    return tf.map_fn(lambda x: tf.image.resize_image_with_crop_or_pad(x, t_height, t_width), image_batch)

# Produces final output of CNN
def inference(o_img, transforms, mode):
    
    nx = CX
    ny = CY

    img_h = tf.shape(o_img)[1]
    img_w = tf.shape(o_img)[2]
    crop_h = img_h - CROP_SIZE
    crop_w = img_w - CROP_SIZE
    
    # Define and apply TPS transform on first image copy
    cp1 = choose_control_points(nx,ny, FLAGS.batch_size, SIGMA_G1, SIGMA2_G1)
    affines = choose_affine_components(FLAGS.batch_size, ROT_1, SCALE_1, TFORM_1)
    T1, fp1 = tps._solve_system(cp1, nx, ny) 
    x_sa, y_sa = tps._transform(T1, fp1, o_img, (img_h,img_w))
    f_img = tps.perform_interpolation(o_img, x_sa, y_sa, (img_h,img_w))
    crop1 = batch_crop(f_img, crop_h, crop_w)    
    
    # Define and apply TPS transform on second image copy
    affines = choose_affine_components(FLAGS.batch_size, ROT_2, SCALE_2, TFORM_2)
    cp2 = choose_control_points(nx, ny, FLAGS.batch_size, SIGMA_G2, SIGMA2_G2)
    T2, fp2 = tps.get_parameter_T(affines, cp2, nx, ny)
    x_sa, y_sa = tps._transform(T2, fp2, f_img, (img_h, img_w))
    f_img2 = tps.perform_interpolation(f_img, x_sa, y_sa, (img_h, img_w))
    
    crop2 = batch_crop(f_img2, crop_h, crop_w)

    img1, img2, T, fp  = tf.cond(mode, lambda: (crop1, crop2, T2, fp2), lambda: (batch_crop(o_img, crop_h, crop_w), crop1, T1, fp1))

    with tf.variable_scope("siamese") as scope:
        o_output = cnn(img1, mode)
        scope.reuse_variables()
        t_output = cnn(img2, mode)

    return o_output, t_output, T, fp


# Returns the overall loss when passed the return values from inference()
def loss(o_output, t_output, transforms, fp, gamma=GAMMA_DEFAULT):
    align = l_align(o_output, transforms, t_output, fp)
    div = gamma * (l_div(o_output) + l_div(t_output))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    # May need to use sum here instead
    return align + div + tf.add_n(reg_losses)

# Train the CNN for one training step
def train(loss, global_step):

    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                          global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    optimizer = tf.train.AdamOptimizer(lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    train_step = optimizer.minimize(loss, global_step)

    with tf.control_dependencies([train_step, variable_averages_op] + update_ops):
        train_op = tf.no_op(name='train')

    return train_op
