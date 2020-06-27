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

import resnet18_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/oran/ILSVRC2012/ILSVRC2012_img_train/',"""Path to the ILSVRC2012 data directory.""")
tf.app.flags.DEFINE_string('weights_dir', '/home/oran/workdir/resnet/full_precision/weights/',"""Path to the Pre-trained weights.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")
tf.app.flags.DEFINE_float('wd', 0.0001,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('dropout', 0.5,"""Dropout rate.""")
tf.app.flags.DEFINE_boolean('projection', True,"""Projection layer or zero padding.""")
tf.app.flags.DEFINE_boolean('use_pretrained', True,"""Whether to use the pre-trained weights or not.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = resnet18_input.IMAGE_SIZE
NUM_CLASSES = resnet18_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = resnet18_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = resnet18_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
delta = 0.15

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 5.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.00001       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def conv_relu(scope, layer_name, prev_layer, conv_shape, stride, train):
  kernel = _variable_with_weight_decay('weights', shape=conv_shape, wd=FLAGS.wd, layer_name=layer_name)
  conv = tf.nn.conv2d(prev_layer, kernel, [1, stride, stride, 1], padding='SAME')
#  biases = _variable_on_cpu('biases', conv_shape[3], tf.constant_initializer(0.0))
#  output = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
  conv_normed = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=train, scope=scope)
  output = tf.nn.relu(conv_normed)
  return output


def res_block(scope, prev_layer, conv_shapes, stride, train):
  # branch2a
  with tf.variable_scope('branch2a') as scope_inner:
      branch2a = conv_relu(scope_inner, scope.name+'_branch2a', prev_layer, conv_shapes[0], stride, train)
  
  # branch2b
  with tf.variable_scope('branch2b') as scope_inner:
      kernel = _variable_with_weight_decay('weights', shape=conv_shapes[1], wd=FLAGS.wd, layer_name=scope.name+'_branch2b')
      conv = tf.nn.conv2d(branch2a, kernel, strides=[1,1,1,1], padding='SAME')
      branch2b = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=train, scope=scope_inner)

  # Input projection and output
  input_depth = prev_layer.get_shape().as_list()[3]
  output_depth = conv_shapes[1][3]
  if (input_depth != output_depth) & (FLAGS.projection==True):
      with tf.variable_scope('branch1') as scope_inner:
          kernel = _variable_with_weight_decay('weights', shape=[1, 1, input_depth, output_depth], wd=FLAGS.wd, layer_name=scope.name+'_branch1')
          branch1 = tf.nn.conv2d(prev_layer, kernel, strides=[1, stride, stride, 1], padding='SAME')
#          branch1 = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=train, scope=scope_inner)
  elif (input_depth != output_depth) & (FLAGS.projection==False):
      with tf.variable_scope('branch1') as scope_inner:
          prev_layer = tf.nn.max_pool(prev_layer, ksize=[1, 1, 1, 1], strides=[1, stride, stride, 1], padding='SAME')
          branch1 = tf.pad(prev_layer, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
  else:
      branch1 = prev_layer
  output = tf.nn.relu(tf.add(branch2b, branch1))
  return output


def _activation_summary(x):
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer, layer_name):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    if (FLAGS.use_pretrained) & (layer_name != 'new'):
        init = np.load(FLAGS.weights_dir+layer_name+'.npy')
        var = tf.get_variable(name, initializer=init, dtype=dtype)
    else:
        init = initializer
        var = tf.get_variable(name, shape, initializer=init, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, wd, layer_name):
  var = _variable_on_cpu(
      name,
      shape,
      tf.contrib.layers.xavier_initializer(uniform=False),
      layer_name)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  images, labels = resnet18_input.distorted_inputs(data_dir=FLAGS.data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
#  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = resnet18_input.inputs(eval_data=eval_data,
                                        data_dir=FLAGS.data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference(images, train=False):
  tf.summary.image('images', images, max_outputs=6)
  ## conv1
  with tf.variable_scope('conv1') as scope:
    conv1 = conv_relu(scope, scope.name, images, [7, 7, 3, 64], 2, train)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

  ## conv2
  # conv2_1
  with tf.variable_scope('conv2_1') as scope:
    conv2_1 = res_block(scope, pool1, [[3,3,64,64],[3,3,64,64]], 1, train)
    _activation_summary(conv2_1)

  # conv2_2
  with tf.variable_scope('conv2_2') as scope:
    conv2_2 = res_block(scope, conv2_1, [[3,3,64,64],[3,3,64,64]], 1, train)
    _activation_summary(conv2_2)

  ## conv3
  # conv3_1
  with tf.variable_scope('conv3_1') as scope:
    conv3_1 = res_block(scope, conv2_2, [[3,3,64,128],[3,3,128,128]], 2, train)
    _activation_summary(conv3_1)

  # conv3_2
  with tf.variable_scope('conv3_2') as scope:
    conv3_2 = res_block(scope, conv3_1, [[3,3,128,128],[3,3,128,128]], 1, train)
    _activation_summary(conv3_2)

  ## conv4  
  # conv4_1
  with tf.variable_scope('conv4_1') as scope:
    conv4_1 = res_block(scope, conv3_2, [[3,3,128,256],[3,3,256,256]], 2, train)
    _activation_summary(conv4_1)
    
  # conv4_2
  with tf.variable_scope('conv4_2') as scope:
    conv4_2 = res_block(scope, conv4_1, [[3,3,256,256],[3,3,256,256]], 1, train)
    _activation_summary(conv4_2)

  ##conv5    
  # conv5_1
  with tf.variable_scope('conv5_1') as scope:
    conv5_1 = res_block(scope, conv4_2, [[3,3,256,512],[3,3,512,512]], 2, train)
    _activation_summary(conv5_1)
    
  # conv5_2
  with tf.variable_scope('conv5_2') as scope:
    conv5_2 = res_block(scope, conv5_1, [[3,3,512,512],[3,3,512,512]], 1, train)
    _activation_summary(conv5_2)

  # pool5
  pool5 = tf.nn.avg_pool(conv5_2, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME', name='pool5')

  # linear layer(WX + b),
  with tf.variable_scope('softmax_linear') as scope:
    pool5 = tf.reshape(pool5, [FLAGS.batch_size, -1])
    weights = _variable_with_weight_decay('weights', [512, NUM_CLASSES], wd=FLAGS.wd, layer_name=scope.name+'_w')
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0), layer_name=scope.name+'_b')
    softmax_linear = tf.add(tf.matmul(pool5, weights), biases, name=scope.name)
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
    np.save(FLAGS.train_dir+'/weights/W_conv1.npy',W)

  with tf.variable_scope('conv2_1', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv2_2', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv3_1', reuse=True) as scope:
      with tf.variable_scope('branch1', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch1.npy',W)
      with tf.variable_scope('branch2a', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv3_2', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv4_1', reuse=True) as scope:
      with tf.variable_scope('branch1', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch1.npy',W)
      with tf.variable_scope('branch2a', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv4_2', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv5_1', reuse=True) as scope:
      with tf.variable_scope('branch1', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch1.npy',W)
      with tf.variable_scope('branch2a', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv5_2', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True) as scope_inner:
        W = quantize(sess)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)
