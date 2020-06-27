from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import resnet18

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/oran/logdir/resnet18_eval',"Directory where to write event logs.")
#tf.app.flags.DEFINE_string('data_dir', '/home/oran/ILSVRC2012/ILSVRC2012_img_val/',"""Either 'test' or 'train_eval'.""")
#tf.app.flags.DEFINE_string('train_dir', '/home/oran/logdir/resnet18_ternary_3',"""Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,"""How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 50000,"""Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,"""Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op, k, W1, W2_1_b2a, W2_1_b2b, W2_2_b2a, W2_2_b2b,
              W3_1_b1, W3_1_b2a, W3_1_b2b, W3_2_b2a, W3_2_b2b,
              W4_1_b1, W4_1_b2a, W4_1_b2b, W4_2_b2a, W4_2_b2b,
              W5_1_b1, W5_1_b2a, W5_1_b2b, W5_2_b2a, W5_2_b2b):
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
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
      if FLAGS.first_layer_binary:
          W1_ = np.load(FLAGS.train_dir+'/weights/W_conv1.npy')
      else:
          W1_ = np.zeros((7,7,3,64))
      W2_1_b2a_ = np.load(FLAGS.train_dir+'/weights/W_conv2_1_branch2a.npy')
      W2_1_b2b_ = np.load(FLAGS.train_dir+'/weights/W_conv2_1_branch2b.npy')
      W2_2_b2a_ = np.load(FLAGS.train_dir+'/weights/W_conv2_2_branch2a.npy')
      W2_2_b2b_ = np.load(FLAGS.train_dir+'/weights/W_conv2_2_branch2b.npy')
      W3_1_b1_ = np.load(FLAGS.train_dir+'/weights/W_conv3_1_branch1.npy')
      W3_1_b2a_ = np.load(FLAGS.train_dir+'/weights/W_conv3_1_branch2a.npy')
      W3_1_b2b_ = np.load(FLAGS.train_dir+'/weights/W_conv3_1_branch2b.npy')
      W3_2_b2a_ = np.load(FLAGS.train_dir+'/weights/W_conv3_2_branch2a.npy')
      W3_2_b2b_ = np.load(FLAGS.train_dir+'/weights/W_conv3_2_branch2b.npy')
      W4_1_b1_ = np.load(FLAGS.train_dir+'/weights/W_conv4_1_branch1.npy')
      W4_1_b2a_ = np.load(FLAGS.train_dir+'/weights/W_conv4_1_branch2a.npy')
      W4_1_b2b_ = np.load(FLAGS.train_dir+'/weights/W_conv4_1_branch2b.npy')
      W4_2_b2a_ = np.load(FLAGS.train_dir+'/weights/W_conv4_2_branch2a.npy')
      W4_2_b2b_ = np.load(FLAGS.train_dir+'/weights/W_conv4_2_branch2b.npy')
      W5_1_b1_ = np.load(FLAGS.train_dir+'/weights/W_conv5_1_branch1.npy')
      W5_1_b2a_ = np.load(FLAGS.train_dir+'/weights/W_conv5_1_branch2a.npy')
      W5_1_b2b_ = np.load(FLAGS.train_dir+'/weights/W_conv5_1_branch2b.npy')
      W5_2_b2a_ = np.load(FLAGS.train_dir+'/weights/W_conv5_2_branch2a.npy')
      W5_2_b2b_ = np.load(FLAGS.train_dir+'/weights/W_conv5_2_branch2b.npy')
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op], 
                               feed_dict={W1: W1_, W2_1_b2a: W2_1_b2a_, W2_1_b2b: W2_1_b2b_, W2_2_b2a: W2_2_b2a_, W2_2_b2b: W2_2_b2b_,
              W3_1_b1:W3_1_b1_, W3_1_b2a:W3_1_b2a_, W3_1_b2b:W3_1_b2b_, W3_2_b2a:W3_2_b2a_, W3_2_b2b:W3_2_b2b_,
              W4_1_b1:W4_1_b1_, W4_1_b2a:W4_1_b2a_, W4_1_b2b:W4_1_b2b_, W4_2_b2a:W4_2_b2a_, W4_2_b2b:W4_2_b2b_,
              W5_1_b1:W5_1_b1_, W5_1_b2a:W5_1_b2a_, W5_1_b2b:W5_1_b2b_, W5_2_b2a:W5_2_b2a_, W5_2_b2b:W5_2_b2b_})
        true_count += np.sum(predictions)
        if step % 40 == 0:        
            summary_str = sess.run(summary_op, feed_dict={W1: W1_, W2_1_b2a: W2_1_b2a_, W2_1_b2b: W2_1_b2b_, W2_2_b2a: W2_2_b2a_, W2_2_b2b: W2_2_b2b_,
                  W3_1_b1:W3_1_b1_, W3_1_b2a:W3_1_b2a_, W3_1_b2b:W3_1_b2b_, W3_2_b2a:W3_2_b2a_, W3_2_b2b:W3_2_b2b_,
                  W4_1_b1:W4_1_b1_, W4_1_b2a:W4_1_b2a_, W4_1_b2b:W4_1_b2b_, W4_2_b2a:W4_2_b2a_, W4_2_b2b:W4_2_b2b_,
                  W5_1_b1:W5_1_b1_, W5_1_b2a:W5_1_b2a_, W5_1_b2b:W5_1_b2b_, W5_2_b2a:W5_2_b2a_, W5_2_b2b:W5_2_b2b_})
            summary_writer.add_summary(summary_str, step)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ %d = %.3f' % (datetime.now(), k, precision))

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  return precision


def evaluate(k=5):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images, labels = resnet18.inputs(eval_data=True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    W1 = tf.placeholder(tf.float32, [7, 7, 3, 64])
    W2_1_b2a = tf.placeholder(tf.float32, [3, 3, 64, 64])
    W2_1_b2b = tf.placeholder(tf.float32, [3, 3, 64, 64])
    W2_2_b2a = tf.placeholder(tf.float32, [3, 3, 64, 64])
    W2_2_b2b = tf.placeholder(tf.float32, [3, 3, 64, 64])
    W3_1_b1 = tf.placeholder(tf.float32, [1, 1, 64, 128])
    W3_1_b2a = tf.placeholder(tf.float32, [3, 3, 64, 128])
    W3_1_b2b = tf.placeholder(tf.float32, [3, 3, 128, 128])
    W3_2_b2a = tf.placeholder(tf.float32, [3, 3, 128, 128])
    W3_2_b2b = tf.placeholder(tf.float32, [3, 3, 128, 128])
    W4_1_b1 = tf.placeholder(tf.float32, [1, 1, 128, 256])
    W4_1_b2a = tf.placeholder(tf.float32, [3, 3, 128, 256])
    W4_1_b2b = tf.placeholder(tf.float32, [3, 3, 256, 256])
    W4_2_b2a = tf.placeholder(tf.float32, [3, 3, 256, 256])
    W4_2_b2b = tf.placeholder(tf.float32, [3, 3, 256, 256])
    W5_1_b1 = tf.placeholder(tf.float32, [1, 1, 256, 512])
    W5_1_b2a = tf.placeholder(tf.float32, [3, 3, 256, 512])
    W5_1_b2b = tf.placeholder(tf.float32, [3, 3, 512, 512])
    W5_2_b2a = tf.placeholder(tf.float32, [3, 3, 512, 512])
    W5_2_b2b = tf.placeholder(tf.float32, [3, 3, 512, 512])
    logits = resnet18.inference(images, W1, W2_1_b2a, W2_1_b2b, W2_2_b2a, W2_2_b2b,
              W3_1_b1, W3_1_b2a, W3_1_b2b, W3_2_b2a, W3_2_b2b,
              W4_1_b1, W4_1_b2a, W4_1_b2b, W4_2_b2a, W4_2_b2b,
              W5_1_b1, W5_1_b2a, W5_1_b2b, W5_2_b2a, W5_2_b2b)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, k)

    # Restore the moving average version of the learned variables for eval.
#    variable_averages = tf.train.ExponentialMovingAverage(
#        resnet18.MOVING_AVERAGE_DECAY)
#    variables_to_restore = variable_averages.variables_to_restore()
#    saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      precision = eval_once(saver, summary_writer, top_k_op, summary_op, k, W1, W2_1_b2a, W2_1_b2b, W2_2_b2a, W2_2_b2b,
              W3_1_b1, W3_1_b2a, W3_1_b2b, W3_2_b2a, W3_2_b2b,
              W4_1_b1, W4_1_b2a, W4_1_b2b, W4_2_b2a, W4_2_b2b,
              W5_1_b1, W5_1_b2a, W5_1_b2b, W5_2_b2a, W5_2_b2b)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
  return precision


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()