from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/oran/logdir/mnist_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
#tf.app.flags.DEFINE_string('train_dir', '/home/oran/logdir/mnist_ternary/4',
#                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op, W1, W2, W_fc, mnist_dataset, images, labels):
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = num_iter * FLAGS.batch_size
    step = 0
    if FLAGS.first_layer_ternary:
        W1_ = np.load(FLAGS.train_dir+'/weights/W_conv1.npy')
    else:
        W1_ = np.zeros((5,5,1,32))
    W2_ = np.load(FLAGS.train_dir+'/weights/W_conv2.npy')
    W_fc_ = np.load(FLAGS.train_dir+'/weights/W_fc.npy')
    while step < num_iter:
      image_batch, label_batch = mnist_dataset.test.next_batch(FLAGS.batch_size)       
      predictions = sess.run([top_k_op], feed_dict={W1: W1_, W2: W2_, W_fc: W_fc_, images: image_batch, labels: label_batch})
      true_count += np.sum(predictions)
      step += 1
#      summary_str = sess.run(summary_op, feed_dict={W1_1: W1_1_, W1_2: W1_2_, W2_1: W2_1_, W2_2: W2_2_, W3_1: W3_1_, W3_2: W3_2_, W_fc: W_fc_}) ## NEW ##
#      summary_writer.add_summary(summary_str, step) ## NEW ##

      # Compute precision @ 1.
    precision = true_count / total_sample_count
    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
  return precision


def evaluate():
  tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  with tf.Graph().as_default() as g:
    mnist_dataset = input_data.read_data_sets(FLAGS.data_dir)
    images = tf.placeholder(tf.float32, [FLAGS.batch_size, 784])
    labels = tf.placeholder(tf.int64, [FLAGS.batch_size])

    # Build a Graph that computes the logits predictions from the
    # inference model.
    W1 = tf.placeholder(tf.float32, [5, 5, 1, 32])
    W2 = tf.placeholder(tf.float32, [5, 5, 32, 64])
    W_fc = tf.placeholder(tf.float32, [3136, 512])
    logits = mnist.inference(images, W1, W2, W_fc)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
#    variable_averages = tf.train.ExponentialMovingAverage(
#        mnist.MOVING_AVERAGE_DECAY)
#    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      precision = eval_once(saver, summary_writer, top_k_op, summary_op, W1, W2, W_fc, mnist_dataset, images, labels)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
  return precision


def main(argv=None):  # pylint: disable=unused-argument
  mnist.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()