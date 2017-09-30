import time
import tensorflow as tf
import lfw_input, lfw_siamese
from datetime import datetime


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_string('train_dir', './lfw_train_model',
                           """Directory where to write event logs """
                           """and checkpoint.""")

# Main training loop
def training():
    with tf.Graph().as_default():
        mode = tf.placeholder(tf.bool)
        global_step = tf.contrib.framework.get_or_create_global_step()

        images, transforms = lfw_input.standardized_input(FLAGS.batch_size)
	
	images = tf.cast(images, tf.float32)

        o_output, t_output, T, fp  = lfw_siamese.inference(images, transforms, mode)

        loss = lfw_siamese.loss(o_output, t_output, T, fp)

        train_step = lfw_siamese.train(loss, global_step)

        # Will trigger after each session run
        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            # Make sure loss is run with train_step
            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time -  self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')

                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks =[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                    tf.train.NanTensorHook(loss),
                    _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=False)) as mon_sess:
	    while not mon_sess.should_stop():
                mon_sess.run(train_step, {mode: True})


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    training()

if __name__ == '__main__':
    with tf.device('/gpu:2'):    
        tf.app.run()

