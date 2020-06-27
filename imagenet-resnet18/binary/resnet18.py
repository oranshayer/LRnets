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
tf.app.flags.DEFINE_string('data_dir_train', '/home/oran/ILSVRC2012/ILSVRC2012_img_train/',"""Path to the ILSVRC2012 data directory.""")
tf.app.flags.DEFINE_string('data_dir_eval', '/home/oran/ILSVRC2012/ILSVRC2012_img_val/',"""Path to the ILSVRC2012 data directory.""")
tf.app.flags.DEFINE_string('weights_dir', '/home/oran/workdir/resnet/full_precision/weights/',"""Path to the Pre-trained weights.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")
tf.app.flags.DEFINE_float('wd', 1e-8,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('wd_weights', 0.00001,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('dropout', 1.0,"""Dropout rate.""")
tf.app.flags.DEFINE_boolean('projection', True,"""Projection layer or zero padding.""")
tf.app.flags.DEFINE_boolean('use_pretrained', False,"""Whether to use the pre-trained weights or not.""")
tf.app.flags.DEFINE_boolean('hot_start', True,"""Whether this is a new run or not.""")
tf.app.flags.DEFINE_string('hot_start_dir', '/home/oran/logdir/resnet18_full_precision/weights',"")
tf.app.flags.DEFINE_float('learning_rate', 0.01,"""Whether this is a new run or not.""")
tf.app.flags.DEFINE_float('lr_decay_epochs', 30,"""Whether this is a new run or not.""")
tf.app.flags.DEFINE_boolean('first_layer_binary', False,"""Whether the first layer  or not.""")
tf.app.flags.DEFINE_boolean('clip_probs', True,"""Whether the clip probabilities or not.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = resnet18_input.IMAGE_SIZE
NUM_CLASSES = resnet18_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = resnet18_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = resnet18_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
delta = 0.15

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


def init_probs(prob):
  init_val = np.array(prob, dtype=np.float32)
  init_val[init_val<1e-4] = 1e-4
  init_val[init_val>0.9999] = 0.9999
  init_val = np.log(init_val/(1-init_val))
  return init_val
  
  
def init_values_hs(W):
    W = np.array(W)
    c_init = 0.55*np.float32(W==0)+0.15 # Was 0.7 , 0.1
    a_nz_init = np.float32(W==1) + 0.5*np.float32(W==0)
    c_init = init_probs(c_init)
    a_nz_init = init_probs(a_nz_init)
    return c_init, a_nz_init
    
    
def init_values_hs_ver2(W):
    W = np.array(W)
    W[W>1] = 1
    W[W<-1] = -1
    a_init = 0.5*(1+W)
    a_init[a_init>0.95] = 0.95
    a_init[a_init<0.05] = 0.05
    a_init = init_probs(a_init)
    return a_init


def initializer(scope, shape, prob):
    if FLAGS.hot_start:
        a_init = init_values_hs_ver2(np.load(FLAGS.hot_start_dir+'/W_'+scope.name.replace('/','_')+'.npy'))
    else:
        a_init = init_probs(np.random.uniform(0.5, 0.5, shape))
    return a_init


def reparametrization(prev_layer, shape, scope, kernel, stride, conv=True, train=True, relu=True):
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
        mu_bar = tf.nn.conv2d(prev_layer, mu,  [1, stride, stride, 1], padding='SAME')
        sigma_bar = tf.sqrt(tf.nn.conv2d(tf.square(prev_layer), var,  [1, stride, stride, 1], padding='SAME')+0.001)
      else:
        mu_bar = tf.matmul(prev_layer, mu)
        sigma_bar = tf.sqrt(tf.matmul(tf.square(prev_layer), var)+0.001)
      res = normal_dist.sample(tf.shape(mu_bar))*sigma_bar + mu_bar
      tf.summary.histogram('a',a)
      tf.summary.histogram('b',b)
  else:
      if conv:
          res = tf.nn.conv2d(prev_layer, kernel, [1, stride, stride, 1], padding='SAME')
      else:
          res = tf.matmul(prev_layer, kernel)
  res_normed = tf.contrib.layers.batch_norm(res, center=True, scale=True, is_training=train, scope=scope)
  if relu: output = tf.nn.relu(res_normed)
  else: output = res_normed
  return output


def get_probs():
  a_ = tf.get_variable('a')
  a = tf.nn.sigmoid(a_)
  return a
  
  
def conv_relu(scope, prev_layer, conv_shape, stride, train):
  kernel = _variable_with_weight_decay(scope.name, shape=conv_shape, wd=FLAGS.wd_weights)
  conv = tf.nn.conv2d(prev_layer, kernel, [1, stride, stride, 1], padding='SAME')
  conv_normed = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=train, scope=scope)
  output = tf.nn.relu(conv_normed)
  return output


def res_block(scope, prev_layer, conv_shapes, stride, train, W1, W2a, W2b):
  # branch2a
  with tf.variable_scope('branch2a') as scope_inner:
      branch2a = reparametrization(prev_layer, conv_shapes[0], scope_inner, W2a, stride, train=train)
  
  # branch2b
  with tf.variable_scope('branch2b') as scope_inner:
      branch2b = reparametrization(branch2a, conv_shapes[1], scope_inner, W2b, 1, train=train, relu=False)

  # Input projection and output
  input_depth = prev_layer.get_shape().as_list()[3]
  output_depth = conv_shapes[1][3]
  if (input_depth != output_depth) & (FLAGS.projection==True):
      with tf.variable_scope('branch1') as scope_inner:
          branch1 = reparametrization(prev_layer, [1, 1, input_depth, output_depth], scope_inner, W1, stride, train=train, relu=False)
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


def _variable_on_cpu(name, shape, initializer):
  if name=='conv1':
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, initializer=initializer, trainable=False)
  else:
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, wd):
  if (FLAGS.first_layer_binary==False) & (name=='conv1'):
      initializer = np.load(FLAGS.hot_start_dir+'/W_'+name+'.npy')
  else:
      initializer = tf.contrib.layers.xavier_initializer(uniform=False)
  var = _variable_on_cpu(name, shape, initializer)
  if (wd is not None) & (name!='conv1'):
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  if not FLAGS.data_dir_train:
    raise ValueError('Please supply a data_dir')
  images, labels = resnet18_input.distorted_inputs(data_dir=FLAGS.data_dir_train,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  if not FLAGS.data_dir_eval:
    raise ValueError('Please supply a data_dir')
  if eval_data: data_dir = FLAGS.data_dir_eval
  else: data_dir = FLAGS.data_dir_train
  images, labels = resnet18_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference(images, W1, W2_1_b2a, W2_1_b2b, W2_2_b2a, W2_2_b2b,
              W3_1_b1, W3_1_b2a, W3_1_b2b, W3_2_b2a, W3_2_b2b,
              W4_1_b1, W4_1_b2a, W4_1_b2b, W4_2_b2a, W4_2_b2b,
              W5_1_b1, W5_1_b2a, W5_1_b2b, W5_2_b2a, W5_2_b2b, train=False):
  tf.summary.image('images', images, max_outputs=1)
  ## conv1
  with tf.variable_scope('conv1') as scope:
    if FLAGS.first_layer_binary:
        conv1 = reparametrization(images, [7, 7, 3, 64], scope, W1, 2, train=train)
    else:
        conv1 = conv_relu(scope, images, [7, 7, 3, 64], 2, train=train)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

  ## conv2
  # conv2_1
  with tf.variable_scope('conv2_1') as scope:
    conv2_1 = res_block(scope, pool1, [[3,3,64,64],[3,3,64,64]], 1, train, None, W2_1_b2a, W2_1_b2b)
    _activation_summary(conv2_1)

  # conv2_2
  with tf.variable_scope('conv2_2') as scope:
    conv2_2 = res_block(scope, conv2_1, [[3,3,64,64],[3,3,64,64]], 1, train, None, W2_2_b2a, W2_2_b2b)
    _activation_summary(conv2_2)

  ## conv3
  # conv3_1
  with tf.variable_scope('conv3_1') as scope:
    conv3_1 = res_block(scope, conv2_2, [[3,3,64,128],[3,3,128,128]], 2, train, W3_1_b1, W3_1_b2a, W3_1_b2b)
    _activation_summary(conv3_1)

  # conv3_2
  with tf.variable_scope('conv3_2') as scope:
    conv3_2 = res_block(scope, conv3_1, [[3,3,128,128],[3,3,128,128]], 1, train, None, W3_2_b2a, W3_2_b2b)
    _activation_summary(conv3_2)

  ## conv4  
  # conv4_1
  with tf.variable_scope('conv4_1') as scope:
    conv4_1 = res_block(scope, conv3_2, [[3,3,128,256],[3,3,256,256]], 2, train, W4_1_b1, W4_1_b2a, W4_1_b2b)
    _activation_summary(conv4_1)
    
  # conv4_2
  with tf.variable_scope('conv4_2') as scope:
    conv4_2 = res_block(scope, conv4_1, [[3,3,256,256],[3,3,256,256]], 1, train, None, W4_2_b2a, W4_2_b2b)
    _activation_summary(conv4_2)

  ##conv5    
  # conv5_1
  with tf.variable_scope('conv5_1') as scope:
    conv5_1 = res_block(scope, conv4_2, [[3,3,256,512],[3,3,512,512]], 2, train, W5_1_b1, W5_1_b2a, W5_1_b2b)
    _activation_summary(conv5_1)
    
  # conv5_2
  with tf.variable_scope('conv5_2') as scope:
    conv5_2 = res_block(scope, conv5_1, [[3,3,512,512],[3,3,512,512]], 1, train, None, W5_2_b2a, W5_2_b2b)
    _activation_summary(conv5_2)

  # pool5
  pool5 = tf.nn.avg_pool(conv5_2, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME', name='pool5')

  # linear layer(WX + b),
  with tf.variable_scope('softmax_linear') as scope:
    pool5 = tf.reshape(pool5, [FLAGS.batch_size, -1])
    weights = _variable_with_weight_decay('weights', [512, NUM_CLASSES], wd=FLAGS.wd_weights)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
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

    
def draw_weights(sess):
  with tf.variable_scope('conv1', reuse=True):
    if FLAGS.first_layer_binary:
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_conv1.npy',W)

  with tf.variable_scope('conv2_1', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv2_2', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv3_1', reuse=True) as scope:
      with tf.variable_scope('branch1', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch1.npy',W)
      with tf.variable_scope('branch2a', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv3_2', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv4_1', reuse=True) as scope:
      with tf.variable_scope('branch1', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch1.npy',W)
      with tf.variable_scope('branch2a', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv4_2', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv5_1', reuse=True) as scope:
      with tf.variable_scope('branch1', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch1.npy',W)
      with tf.variable_scope('branch2a', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

  with tf.variable_scope('conv5_2', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2a.npy',W)
      with tf.variable_scope('branch2b', reuse=True):
        a = get_probs()
        a_ = sess.run(a)
        W = draw_ternary_weight(a_)
        np.save(FLAGS.train_dir+'/weights/W_'+scope.name+'_branch2b.npy',W)

def clip_probs():
  with tf.variable_scope('conv1', reuse=True):
    if FLAGS.first_layer_binary:
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip)

  with tf.variable_scope('conv2_1', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip) # Fix for first layer binary
      with tf.variable_scope('branch2b', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)

  with tf.variable_scope('conv2_2', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)
      with tf.variable_scope('branch2b', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)

  with tf.variable_scope('conv3_1', reuse=True) as scope:
      with tf.variable_scope('branch1', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)
      with tf.variable_scope('branch2a', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)
      with tf.variable_scope('branch2b', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)

  with tf.variable_scope('conv3_2', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)
      with tf.variable_scope('branch2b', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)

  with tf.variable_scope('conv4_1', reuse=True) as scope:
      with tf.variable_scope('branch1', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)
      with tf.variable_scope('branch2a', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)
      with tf.variable_scope('branch2b', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)

  with tf.variable_scope('conv4_2', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)
      with tf.variable_scope('branch2b', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)

  with tf.variable_scope('conv5_1', reuse=True) as scope:
      with tf.variable_scope('branch1', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)
      with tf.variable_scope('branch2a', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)
      with tf.variable_scope('branch2b', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)

  with tf.variable_scope('conv5_2', reuse=True) as scope:
      with tf.variable_scope('branch2a', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)
      with tf.variable_scope('branch2b', reuse=True):
        a_ = tf.get_variable('a')
        clip = tf.assign(a_, tf.clip_by_value(a_, -4.6, 4.6))
        clip_op = tf.group(clip_op, clip)
  return clip_op