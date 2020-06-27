from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import numpy as np

import mnist_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 256,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/oran/logdir/mnist_data',"""Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_string('hot_start_dir', '/home/oran/logdir/mnist_full_precision',"""Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_float('wd', 1e-6,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('wd_weights', 0.0001,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('dropout', 0.5,"""Dropout rate.""")
tf.app.flags.DEFINE_boolean('hot_start', True,"""Whether this is a new run or not.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01,"""Whether this is a new run or not.""")
tf.app.flags.DEFINE_float('lr_decay_epochs', 100,"""Whether this is a new run or not.""")
tf.app.flags.DEFINE_boolean('first_layer_ternary', True,"""Whether the first layer  or not.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = mnist_input.IMAGE_SIZE
NUM_CLASSES = mnist_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = mnist_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = mnist_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def draw_ternary_weight(a):
    var = 2*(np.random.binomial(1,p=a) - 0.5)
    return var
    
    
def init_values_hs(mu_):
    c_init = np.array(mu_)
    c_init = 1 - np.abs(c_init)
    c_init[c_init>0.9999] = 0.9999
    c_init = np.log(c_init/(1-c_init))
    a_nz_init = np.array(mu_)
    a_nz_init[a_nz_init<1e-4] = 1e-4
    a_nz_init[a_nz_init>1e-4] = 0.9999
    a_nz_init = np.log(a_nz_init/(1-a_nz_init))
    return c_init, a_nz_init
    
    
def init_values_hs_ver2(W):
    W = np.array(W)
    W[W>0] = 1
    W[W<=0] = -1
    c_init = 0.55*np.float32(W==0)+0.15 # Was 0.7 , 0.1
    a_nz_init = 0.6*np.float32(W==1) + 0.2
    c_init = init_probs(c_init)
    a_nz_init = init_probs(a_nz_init)
    return c_init, a_nz_init
    
    
def init_values_hs_ver3(W):
    W = np.array(W)
    W[W>1] = 1
    W[W<-1] = -1
    a_init = 0.5*(1+W)
    a_init[a_init>0.95] = 0.95
    a_init[a_init<0.05] = 0.05
    a_init = init_probs(a_init)
    return a_init
   
   
def init_probs(prob):
  init_val = np.array(prob, dtype=np.float32)
  init_val[init_val<1e-4] = 1e-4
  init_val[init_val>0.9999] = 0.9999
  init_val = np.log(init_val/(1-init_val))
  return init_val
  

def initializer(scope, shape, prob):
    if FLAGS.hot_start:
        a_init = init_values_hs_ver3(np.load(FLAGS.hot_start_dir+'/W_'+scope.name+'.npy'))
    else:
        a_init = init_probs(np.random.uniform(0.5, 0.5, shape))
    return a_init

    
def reparametrization(prev_layer, shape, scope, kernel, conv=True, train=True):
  a_ = tf.get_variable('a', initializer=initializer(scope, shape, 'a'), dtype=tf.float32)
#  a_ = tf.get_variable('a_nz', shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False), dtype=tf.float32)

  a = tf.nn.sigmoid(a_)
  b = 1-a
  beta_loss = tf.multiply(tf.reduce_sum(tf.multiply(a, 1-a)), FLAGS.wd, name='beta_loss_a')
  tf.add_to_collection('losses', beta_loss)
  if train:
      mu = a - b
      var = a + b - tf.square(mu)
      normal_dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
      if conv:
        mu_bar = tf.nn.conv2d(prev_layer, mu,  [1, 1, 1, 1], padding='SAME')
        sigma_bar = tf.sqrt(tf.nn.conv2d(tf.square(prev_layer), var,  [1, 1, 1, 1], padding='SAME')+0.001)
      else:
        mu_bar = tf.matmul(prev_layer, mu)
        sigma_bar = tf.sqrt(tf.matmul(tf.square(prev_layer), var)+0.001)
      res = normal_dist.sample(tf.shape(mu_bar))*sigma_bar + mu_bar
      tf.summary.histogram('a',a)
      tf.summary.histogram('b',b)
  else:
      if conv:
          res = tf.nn.conv2d(prev_layer, kernel, [1, 1, 1, 1], padding='SAME')
      else:
          res = tf.matmul(prev_layer, kernel)
  res_normed = tf.contrib.layers.batch_norm(res, center=True, scale=True, is_training=train)
  output = tf.nn.relu(res_normed)
  return output
  
def get_probs():
  a_ = tf.get_variable('a')
  a = tf.nn.sigmoid(a_)
  return a

def conv_relu(scope, prev_layer, conv_shape, train):
  kernel = _variable_with_weight_decay(scope.name, shape=conv_shape, wd=FLAGS.wd_weights)
  conv = tf.nn.conv2d(prev_layer, kernel, [1, 1, 1, 1], padding='SAME')
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
  if name=='conv1_1':
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, initializer=initializer, trainable=False)
  else:
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, wd):
  if (FLAGS.first_layer_ternary==False) & (name=='conv1_1'):
      initializer = np.load(FLAGS.hot_start_dir+'/W_'+name+'.npy')
  else:
      initializer = tf.contrib.layers.xavier_initializer(uniform=False)
  var = _variable_on_cpu(name, shape, initializer)
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
  return images, labels


def inputs(eval_data):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = mnist_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  return images, labels


def inference(images, W1, W2, W_fc, train=False):
  images_reshaped = tf.reshape(images, [FLAGS.batch_size, 28, 28, 1])
  # conv1
  with tf.variable_scope('conv1') as scope:
    if FLAGS.first_layer_ternary:
        conv1 = reparametrization(images_reshaped, [5, 5, 1, 32], scope, W1, train=train)
    else:
        conv1 = conv_relu(scope, images_reshaped, [5, 5, 1, 32], train=train)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    conv2 = reparametrization(pool1, [5, 5, 32, 64], scope, W2, train=train)
    _activation_summary(conv2)

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # fc
  with tf.variable_scope('fc') as scope:
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    fc = reparametrization(reshape, [dim, 512], scope, W_fc, conv=False, train=train)
    if train:
        fc = tf.nn.dropout(fc, FLAGS.dropout)
    _activation_summary(fc)

  # Linear classifier
  with tf.variable_scope('softmax_linear'):
    weights = _variable_with_weight_decay('weights', [512, NUM_CLASSES], wd=FLAGS.wd_weights)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(fc, weights), biases)
    _activation_summary(softmax_linear)

  return softmax_linear
  

def draw_weights(sess):
  with tf.variable_scope('conv1', reuse=True) as scope:
    if FLAGS.first_layer_ternary:
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'.npy',W)
    
  with tf.variable_scope('conv2', reuse=True) as scope:
    a = get_probs()
    a_ = sess.run(a)
    W = draw_ternary_weight(a_)
    np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'.npy',W)
    
  with tf.variable_scope('fc', reuse=True) as scope:
    a = get_probs()
    a_ = sess.run(a)
    W = draw_ternary_weight(a_)
    np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'.npy',W)


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

