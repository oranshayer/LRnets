from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import numpy as np

from six.moves import urllib
import tensorflow as tf

import mnist_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 256,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/oran/logdir/mnist_data',"""Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")
tf.app.flags.DEFINE_float('wd', 0.0001,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('dropout', 0.5,"""Dropout rate.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = mnist_input.IMAGE_SIZE
NUM_CLASSES = mnist_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = mnist_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = mnist_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
delta = 0.15

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 50.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def conv_relu(scope, name, prev_layer, conv_shape, train):
  kernel = _variable_with_weight_decay('weights', shape=conv_shape, wd=FLAGS.wd)
  conv = tf.nn.conv2d(prev_layer, kernel, [1, 1, 1, 1], padding='SAME')
#  biases = _variable_on_cpu('biases', conv_shape[3], tf.constant_initializer(0.0))
#  output = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
  conv_normed = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=train, scope=scope)
  output = tf.nn.relu(conv_normed, name=scope.name)
  return output

def _activation_summary(x):
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, wd):
  var = _variable_on_cpu(
      name,
      shape,
      tf.contrib.layers.xavier_initializer(uniform=False))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = mnist_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = mnist_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference(images, train=False):
  images_reshaped = tf.reshape(images, [FLAGS.batch_size, 28, 28, 1])
  # conv1
  with tf.variable_scope('conv1') as scope:
    conv1 = conv_relu(scope, scope.name, images_reshaped, [5, 5, 1, 32], train)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    conv2 = conv_relu(scope, scope.name, pool1, [5, 5, 32, 64], train)
    _activation_summary(conv2)

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # fc
  with tf.variable_scope('fc') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 512], wd=FLAGS.wd)
    fc_ = tf.matmul(reshape, weights)
    fc_normed = tf.contrib.layers.batch_norm(fc_, center=True, scale=True, is_training=train, scope=scope)
    fc = tf.nn.relu(fc_normed, name=scope.name)
    if train:
        fc = tf.nn.dropout(fc, FLAGS.dropout)
    _activation_summary(fc)

  # Classifier
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [512, NUM_CLASSES], wd=FLAGS.wd)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(fc, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def loss(logits, labels):
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def quantize(sess):    
    if False:
        kernel_ = tf.get_variable('weights')
        kernel = sess.run(kernel_)
        kernel = kernel/(np.std(kernel))
        kernel[kernel>delta] = 1
        kernel[(kernel>=-delta) & (kernel <=delta)] = 0
        kernel[kernel<-delta] = -1
    else:
        kernel_ = tf.get_variable('weights')
        kernel = sess.run(kernel_)
        kernel = kernel/(np.std(kernel))
    return kernel
    
    
def save_weights(sess):
  with tf.variable_scope('conv1', reuse=True):
    W = quantize(sess)
    np.save(FLAGS.train_dir+'/W_conv1.npy',W)

  with tf.variable_scope('conv2', reuse=True):
    W = quantize(sess)
    np.save(FLAGS.train_dir+'/W_conv2.npy',W)

  with tf.variable_scope('fc', reuse=True):
    W = quantize(sess)
    np.save(FLAGS.train_dir+'/W_fc.npy',W)
