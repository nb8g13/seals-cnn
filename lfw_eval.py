import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import math
import time
import random

from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from scipy.ndimage.interpolation import affine_transform
from sklearn.neighbors import KDTree
from skimage.color import rgb2gray

import feature_plotting as fp
import numpy as np
from PIL import Image

import lfw_siamese
import lfw_input

# Number of eigenvectors to keep from the PDM on landmarks
LANDMARK_LIMIT = 128
# Number of eigenvectors to keep from the PDM on images patchs around each landmark
PATCH_LIMIT = 128
# Number of landmarks to keep from the PDM built from combining the PDM of patches and landmarks
OVERALL_LIMIT = 128
# Size of images patches used in the PDM of image patches
EIG_PATCH = 5
# Max number of tries for procurstes analysis on image landmarks
MAX_TRIES = 100
# Threshold for procrustes analysis
THRESHOLD = 0.1

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 19962,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                             """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './lfw_train_model', '''Checkpoint directory''')

# Creates heatmaps of each probability map output by the CNN and stores them in a directory
def find_heatmap(o_output):
    shape = o_output.shape
    im_no = 0
    while im_no < shape[0]:
        landmark = 0
        while landmark < shape[3]:
            plt.imsave("heatmaps/output-%d-map-%d.tiff" % (im_no, landmark),np.log(o_output[im_no, :, :, landmark]+1e-8), cmap="hot")
            landmark += 1
        im_no += 1

# Evaluate one batch
def eval_once(saver, indice_op, img_batch, mode, o_output):
    general_saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            general_saver = general_saver.restore(sess, ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
	    print("printing global step")
	    print(global_step)
        else:
            print('No checkpoint file found')
            return

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter  = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            step = 0
            im_saver = fp.ImageSaver()
            count = 0

            full_results = None

            while step < num_iter and not coord.should_stop():
                results = sess.run([indice_op, img_batch, o_output], {mode:False})
                landmarks = results[0]
                imgs = results[1]
                maps = results[2]
                #warps = results[3] # aff img no longer evaluated
		step = step + 1
                
                if full_results is None:
                    full_results = landmarks

                else:
                    full_results = np.concatenate((full_results, landmarks))
                              
                if count == 0:
                    find_heatmap(maps)
                for i in range(0, FLAGS.batch_size):
                    im_saver.save_image(imgs[i, :, : , :], count, landmarks[i, :, :])
                    count = count + 1
                    # plt.imsave("warped-images/output-%d.tiff" % count, warps[i, :,:,:])
            
            # Construct and test the PDM models
            # Find means
            means = np.mean(full_results, axis=1, keepdims=True)
            # Substract means
            centered_results  = full_results - means
            # Get norms
            norms = np.linalg.norm(centered_results, axis=(1,2), keepdims = True)
            # Divide through by norms
            unit_results = centered_results / norms
            # Get random regerence shape
            reference_shape = np.squeeze(unit_results[random.randrange(unit_results.shape[0]),:,:])
            
            # Set disparity to unreachable number
            total_disparity = float("inf")
            # Record number of procrustes loops and set maximum number of tries
            tries = 0
            max_tries = MAX_TRIES
            # Set acceptance threshold
            threshold = THRESHOLD
            # Until we exceed the try limit
            while(tries < max_tries) :
                rotations = None
                scales = None
                updated_coords = None
                total_disparity = 0
                # For each set of coordinates
                for i in range(0,unit_results.shape[0]):
                    # Get the coordinate and do procrustes
                    coords = np.squeeze(unit_results[i,:,:])
                    R, s = orthogonal_procrustes(reference_shape, coords)
                    # Record the rotation and scale variables 
                    if rotations is None:
                        rotations = np.expand_dims(R, 0)
                        scales = np.expand_dims(s, 0)
                    else:
                        rotations = np.concatenate((rotations, np.expand_dims(R,0)))
                        scales = np.concatenate((scales, np.expand_dims(s, 0)))
                    new_coords = np.dot(coords, R.T) * s
                    if updated_coords is None:
                        updated_coords = np.expand_dims(new_coords, 0)
                    else:
                        updated_coords = np.concatenate((updated_coords, np.expand_dims(new_coords, 0)))
                    disparity = np.sum(np.square(reference_shape - new_coords))
                    total_disparity += disparity
                total_disparity = total_disparity / unit_results.shape[0]
                if total_disparity < threshold:
                    break
                else:
                    print("Threshold too high making next attempt")
                    tries += 1
                    reference_shape = np.squeeze(np.mean(updated_coords, axis=0))
                    ref_mean = np.mean(reference_shape, axis=0)
                    ref_norm = np.linalg.norm(reference_shape)
                    reference_shape -= ref_mean
                    reference_shape /= ref_norm     
            updated_norms = updated_coords * norms
            updated_means = updated_norms + means
            patch_matrix = None
            
            for i in range(0, unit_results.shape[0]):
                r = rotations[i,:,:]
                s = scales[i]
                img = np.array(Image.open("clean-images/output-%d.tiff" % i).convert("L"))
                t_img = interpolate(img.shape[0], img.shape[1], norms[i,:], means[i,:], r, s, img.astype(np.float32))
                new_img = Image.fromarray(np.uint8(t_img)).save("t-images/output-%d.tiff" % i)
                patch_list = None
                for j in range(0, updated_means.shape[1]):
                    landmark = np.squeeze(updated_means[i,j,:])
                    lidx = landmark.astype(np.int64)           
                    patch_pad = EIG_PATCH / 2 
                    patch = t_img[lidx[0]-patch_pad:lidx[0]+patch_pad+1, lidx[1]-patch_pad:lidx[1]+patch_pad+1].flatten()
                    if patch_list is None:
                        patch_list = patch
                    else:
                        np.concatenate((patch_list, patch))
                patch_list = patch_list
                if patch_matrix is None:
                    patch_matrix = np.expand_dims(patch_list, 0)
                else:
                    patch_matrix = np.concatenate((patch_matrix, np.expand_dims(patch_list,0)), 0)
            
            patch_mean = np.mean(patch_matrix, 0, keepdims=True)
            patch_matrix = patch_matrix - patch_mean
            patch_sq = np.dot(np.transpose(patch_matrix), patch_matrix)
            pvals, pvecs = np.linalg.eig(patch_sq)
            idx = pvals.argsort()[::-1]
            pvals = pvals[idx]
            pvecs = pvecs[:, idx]
            pvecs = pvecs[:, :LANDMARK_LIMIT]
            p_projections = np.dot(patch_matrix, pvecs)
            
            coord_matrix = np.reshape(updated_coords, (unit_results.shape[0], -1))
            coord_sq = np.dot(np.transpose(coord_matrix), coord_matrix)
            cvals, cvecs = np.linalg.eig(coord_sq)
            idx = cvals.argsort()[::-1]
            cvals = cvals[idx]
            cvecs = cvecs[:, idx]
            cvecs = cvecs[:,:PATCH_LIMIT]
	    c_projections = np.dot(coord_matrix, cvecs)

            total_projections = np.concatenate((p_projections, c_projections), axis=1)
            total_sq = np.dot(np.transpose(total_projections), total_projections)
            tvals, tvecs = np.linalg.eig(total_sq)
            idx = tvals.argsort()[::-1]
            tvals = tvals[idx]
            tvecs = tvecs[:, idx]
            tvecs = tvecs[:,:OVERALL_LIMIT]
            t_projections = np.dot(total_projections, tvecs)

            search_tree = KDTree(t_projections)
            print("Printing 5 best matches for first 10 images in output-images")
            for i in range(0,10):
                dist, ind = search_tree.query(np.expand_dims(t_projections[i,:],0), k=5)
                print(ind)
                           
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

# Takes the estimated transform from procrustes and performs it on each image, so that image
# patches can be retreived from the mapped landmarks - very similar to the interpolate function in
# the TPS module, but uses numpy instead of tensorflow
def interpolate(height, width, norm, mean, r, s, img):
     
    flat_im = img.flatten()

    h = np.expand_dims(np.arange(0, height, dtype=np.float32),1)
    h = np.repeat(h, width, axis = 1).flatten()
    
    w = np.expand_dims(np.arange(0, width, dtype=np.float32),1)
    w = np.repeat(w, height, axis=1).T.flatten()
    
    coords = np.stack([h,w], axis=1)
    coords -= mean
    coords /= norm

    coords_t = np.dot(coords, r.T) * s
    
    coords_t *= norm
    coords_t += mean    
    coord_0 = np.floor(coords_t).astype(np.int32)
    coord_1 = coord_0 + 1

    x = coords_t[:,0]
    y = coords_t[:,1]

    x0 = coord_0[:,0]
    x1 = coord_1[:,0]
    y0 = coord_0[:,1]
    y1 = coord_1[:,1]

    x0 = np.clip(x0, 0, height-1)
    x1 = np.clip(x1,0, height-1)
    y0 = np.clip(y0, 0, width-1)
    y1 = np.clip(y1, 0, width-1)

    idx_a = width*x0 + y0
    idx_b = width*x1 + y0
    
    idx_c = width*x0 + y1
    idx_d = width*x1 + y1

    Ia = np.take(flat_im, idx_a)
    Ib = np.take(flat_im, idx_b)
    Ic = np.take(flat_im, idx_c)
    Id = np.take(flat_im, idx_d)

    x0_f = x0.astype(np.float32)
    x1_f = x1.astype(np.float32)
    y0_f = y0.astype(np.float32)
    y1_f = y1.astype(np.float32)

    wa = (x1_f - x) * (y1_f - y)
    wb = (x1_f - x) * (y - y0_f)
    wc = (x - x0_f) * (y1_f - y)
    wd = (x - x0_f) * (y - y0_f)

    output = wa*Ia + wb*Ib + wc*Ic + wd*Id
    
    return np.reshape(output, (height, width))

# Main evaluation loop
def evaluate():
    print('in evaluate()')
    with tf.Graph().as_default() as g:
        eval_data = FLAGS.eval_data == "test"

	mode = tf.placeholder(tf.bool)

        img_batch, transform_batch = lfw_input.normal_inputs(eval_data, FLAGS.batch_size)

	img = tf.cast(img_batch, tf.float32)

        o_output, t_output, T, fp  = lfw_siamese.inference(img, transform_batch, mode)

        indice_op = lfw_siamese.find_expected_indices(o_output)

        variable_averages = tf.train.ExponentialMovingAverage(
            lfw_siamese.MOVING_AVERAGE_DECAY)
        print("loading vars")
        variables_to_restore = variable_averages.variables_to_restore()
        print('Variables restored')
        saver = tf.train.Saver(variables_to_restore)

        eval_once(saver, indice_op, img_batch, mode, o_output)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

if __name__ == "__main__":
    tf.app.run()



