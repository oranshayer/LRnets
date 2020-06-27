from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/oran/logdir/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
#tf.app.flags.DEFINE_string('train_dir', '/home/oran/logdir/cifar10_ternary/4',
#                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op, W1_1, W1_2, W2_1, W2_2, W3_1, W3_2, W_fc):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      if FLAGS.first_layer_ternary:
          W1_1_ = np.load(FLAGS.train_dir+'/weights/W_conv1_1.npy')
      else:
          # The FP weights are loaded from the model, doesn't matter what we input in the PH
          W1_1_ = np.zeros((3,3,3,128))
      W1_2_ = np.load(FLAGS.train_dir+'/weights/W_conv1_2.npy')
      W2_1_ = np.load(FLAGS.train_dir+'/weights/W_conv2_1.npy')
      W2_2_ = np.load(FLAGS.train_dir+'/weights/W_conv2_2.npy')
      W3_1_ = np.load(FLAGS.train_dir+'/weights/W_conv3_1.npy')
      W3_2_ = np.load(FLAGS.train_dir+'/weights/W_conv3_2.npy')
      W_fc_ = np.load(FLAGS.train_dir+'/weights/W_fc.npy')
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op], feed_dict={W1_1: W1_1_, W1_2: W1_2_, W2_1: W2_1_, W2_2: W2_2_, W3_1: W3_1_, W3_2: W3_2_, W_fc: W_fc_})
        true_count += np.sum(predictions)
        step += 1
#        summary_str = sess.run(summary_op, feed_dict={W1_1: W1_1_, W1_2: W1_2_, W2_1: W2_1_, W2_2: W2_2_, W3_1: W3_1_, W3_2: W3_2_, W_fc: W_fc_}) ## NEW ##
#        summary_writer.add_summary(summary_str, step) ## NEW ##

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

#      summary = tf.Summary()
#      summary.ParseFromString(sess.run(summary_op, feed_dict={W1: W1_, W2: W2_, W5: W5_}))        
#      summary.value.add(tag='Precision @ 1', simple_value=precision)
#      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  return precision


def evaluate():
  tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    W1_1 = tf.placeholder(tf.float32, [3, 3, 3, 128])
    W1_2 = tf.placeholder(tf.float32, [3, 3, 128, 128])
    W2_1 = tf.placeholder(tf.float32, [3, 3, 128, 256])
    W2_2 = tf.placeholder(tf.float32, [3, 3, 256, 256])
    W3_1 = tf.placeholder(tf.float32, [3, 3, 256, 512])
    W3_2 = tf.placeholder(tf.float32, [3, 3, 512, 512])
    W_fc = tf.placeholder(tf.float32, [8192, 1024])
    logits = cifar10.inference(images, W1_1, W1_2, W2_1, W2_2, W3_1, W3_2, W_fc)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
#    variable_averages = tf.train.ExponentialMovingAverage(
#        cifar10.MOVING_AVERAGE_DECAY)
#    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      precision = eval_once(saver, summary_writer, top_k_op, summary_op, W1_1, W1_2, W2_1, W2_2, W3_1, W3_2, W_fc)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
  return precision


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()