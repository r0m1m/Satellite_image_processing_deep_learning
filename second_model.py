from tricks import *
import sys
import os

nclasses=6

def myModel(x):

  conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[5,5], padding="valid", 
                           activation=tf.nn.relu)
  conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=[3,3], padding="valid", 
                           activation=tf.nn.relu)
  conv3 = tf.layers.conv2d(inputs=conv2, filters=16, kernel_size=[3,3], padding="valid", 
                           activation=tf.nn.relu)
  conv4 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=[3,3], padding="valid",
                           activation=tf.nn.relu)
  conv5 = tf.layers.conv2d(inputs=conv4, filters=32, kernel_size=[3,3], padding="valid",
                           activation=tf.nn.relu)
  conv6 = tf.layers.conv2d(inputs=conv5, filters=32, kernel_size=[3,3], padding="valid",
                           activation=tf.nn.relu)
  conv7 = tf.layers.conv2d(inputs=conv6, filters=32, kernel_size=[2,2], padding="valid",
                           activation=tf.nn.relu)

  features = tf.reshape(conv7, shape=[-1, 32], name="features")

  estimated = tf.layers.dense(inputs=features, units=nclasses, activation=None)
  estimated_label = tf.argmax(estimated, 1, name="prediction")

  return estimated, estimated_label

if len(sys.argv) != 2:
  print("Usage : <output directory for SavedModel>")
  sys.exit(1)

with tf.Graph().as_default():
  
  x = tf.placeholder(tf.float32, [None, None, None, 4], name="x")
  y = tf.placeholder(tf.int32  , [None, None, None, 1], name="y")
  lr = tf.placeholder_with_default(tf.constant(0.0002, dtype=tf.float32, shape=[]),
                                   shape=[], name="lr")
  
  y_estimated, y_label = myModel(x)
  
  cost = tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(y, [-1, 1]), 
                                                logits=tf.reshape(y_estimated, [-1, nclasses]))
  
  optimizer = tf.train.AdamOptimizer(learning_rate=lr, name="optimizer").minimize(cost)
  
  init = tf.global_variables_initializer()
  saver = tf.train.Saver( max_to_keep=20 )
  sess = tf.Session()
  sess.run(init)

  create_savedmodel(sess, ["x:0", "y:0"], ["features:0", "prediction:0"], sys.argv[1])