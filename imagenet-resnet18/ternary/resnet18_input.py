from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 224

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1281167
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 50000


def compute_bigger_dim(smaller, bigger, new_smaller_dim):
    return tf.cast(bigger*new_smaller_dim/smaller, tf.int32)


def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)
  
  filename = tf.string_split([key],'/')
  filename = filename.values[4]
  label = tf.string_split([filename],'_')
  label = label.values[0]
  label = tf.string_to_number(label, out_type=tf.int32)
  result.label = tf.reshape(label,[1,])

  result.uint8image = tf.image.decode_jpeg(value, channels=3)

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 10 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 10 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.
  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames_ = os.listdir(data_dir)
  filenames = [os.path.join(data_dir, filenames_[i])
               for i in xrange(0, len(filenames_))]
#  for f in filenames:
#    if not tf.gfile.Exists(f):
#      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
#  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  reshaped_image = tf.image.convert_image_dtype(read_input.uint8image, tf.float32)
  # Random resize -- new 23.8
  shape = tf.shape(reshaped_image)
  h = shape[0]
  w = shape[1]
  new_size = tf.random_uniform([], minval=256, maxval=257, dtype=tf.int32)
  new_height_and_width = tf.cond(h < w,
    lambda: (new_size, compute_bigger_dim(h, w, new_size)),
    lambda: (compute_bigger_dim(w, h, new_size), new_size))
  reshaped_image = tf.image.resize_images(reshaped_image, [new_height_and_width[0], new_height_and_width[1]])

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  # NOTE: since per_image_standardization zeros the mean and makes
  # the stddev unit, this likely has no effect see tensorflow#1458.
#  distorted_image = tf.image.random_brightness(distorted_image,
#                                               max_delta=0.4) # was 63
#  distorted_image = tf.image.random_contrast(distorted_image,
#                                             lower=0.6, upper=1.4) # was 0.2, 1.8

  # Subtract off the mean and divide by the variance of the pixels.
#  float_image = tf.image.per_image_standardization(distorted_image)
  imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
  float_image = (distorted_image - imagenet_mean) / imagenet_std
  
  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.0025
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames_ = os.listdir(data_dir)
  filenames = [os.path.join(data_dir, filenames_[i])
               for i in xrange(0, len(filenames_))]
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

#  for f in filenames:
#    if not tf.gfile.Exists(f):
#      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
#  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  reshaped_image = tf.image.convert_image_dtype(read_input.uint8image, tf.float32)
  shape = tf.shape(reshaped_image)
  h = shape[0]
  w = shape[1]
  new_size = tf.random_uniform([], minval=256, maxval=257, dtype=tf.int32)
  new_height_and_width = tf.cond(h < w,
    lambda: (new_size, compute_bigger_dim(h, w, new_size)),
    lambda: (compute_bigger_dim(w, h, new_size), new_size))
  reshaped_image = tf.image.resize_images(reshaped_image, [new_height_and_width[0], new_height_and_width[1]])

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         height, width)

  # Subtract off the mean and divide by the variance of the pixels.
#  float_image = tf.image.per_image_standardization(resized_image)
  imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
  float_image = (resized_image - imagenet_mean) / imagenet_std

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.02
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)