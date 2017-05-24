
# %% Borrowed utils from here: https://github.com/pkmital/tensorflow_tutorials/
import tensorflow as tf
import numpy as np

def conv2d(x, n_filters,
           k_h=5, k_w=5,
           stride_h=2, stride_w=2,
           stddev=0.02,
           activation=lambda x: x,
           bias=True,
           padding='SAME',
           name="Conv2D"):
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w', [k_h, k_w, x.get_shape()[-1], n_filters],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(
            x, w, strides=[1, stride_h, stride_w, 1], padding=padding)
        if bias:
            b = tf.get_variable(
                'b', [n_filters],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = conv + b
        return conv
    
def linear(x, n_units, scope=None, stddev=0.02,
           activation=lambda x: x):
    shape = x.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], n_units], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        return activation(tf.matmul(x, matrix))
    
# %%
def weight_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)

# %%
def bias_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)

# %% 
def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot
