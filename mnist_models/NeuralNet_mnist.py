#### =========================================================================================
#### =========================================================================================
###### Neural Network Example with tensorflow
######
#### =========================================================================================
#### =========================================================================================


import tensorflow as tf
import numpy as np 
from keras.datasets import mnist

#### =========================================================================================
## Part 1: Load in Data
#### =========================================================================================
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],-1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# one hot encode y_train
a = y_train
b = np.zeros((y_train.shape[0],10))
b[np.arange(y_train.shape[0]), a] = 1
y_train = b
# one hot encode y_test
a = y_test
b = np.zeros((y_test.shape[0],10))
b[np.arange(y_test.shape[0]), a] = 1
y_test = b

print "X_train shape: {}".format(X_train.shape)
print "X_test shape: {}".format(X_test.shape)
print "y_train shape:{}".format(y_train.shape)
print "y_test shape:{}".format(y_test.shape)


#### =========================================================================================
##  Part 1: Define graph architecture
#### =========================================================================================

def layer(input_, weight_shape, bias_shape):
	"""Defines hidden layer of neural network"""
	weight_stdv = 2.0 / weight_shape[0]
	weight_init = tf.random_normal_initializer(mean=0, stddev=weight_stdv)
	bias_init = tf.constant_initializer(value=0)
	W = tf.get_variable("W", shape=weight_shape, initializer=weight_init)
	bias = tf.get_variable("bias", shape=bias_shape, initializer=bias_init)
	return tf.nn.relu(tf.matmul(input_,W) + bias)

def feed_forward(x):
	"""Feed Forward Neural Network"""
	with tf.variable_scope('hidden_1'):
		output_1 = layer(x,[784,100],[100])
	with tf.variable_scope('hidden_2'):
		output_2 = layer(output_1, [100,50],[50])
	with tf.variable_scope('hidden_3'):
		output_3 = layer(output_2, [50,10],[10])
	return output_3

# def loss(output, y):
# 	"""Calculate cross-entropy loss of network"""
# 	logprod = y * tf.log(output)
# 	xentropy = -tf.reduce_sum(logprod, axis=1)
# 	total_loss = tf.reduce_mean(xentropy)
# 	return total_loss

def loss(output, y):
 xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
 loss = tf.reduce_mean(xentropy)
 return loss

def evaluate(output, y):
	"""Calculates accuracy of model"""
	# get predicted classes 
	preds = tf.argmax(output,1) 
	# get actual classes
	correct = tf.argmax(y,1)
	# number equal
	eq = tf.equal(preds, correct)
	return tf.reduce_mean(tf.cast(eq, tf.float32))

def train(cost, global_step):
	"""trains model"""
	tf.summary.scalar("cost", cost)
	optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
	train_op = optimizer.minimize(cost, global_step=global_step)
	return train_op


learning_rate = 0.01
epochs = 100
display_step = 10
batch_size = 32
momentum = 0.9

# train model
with tf.Graph().as_default():

	# init inputs
	x = tf.placeholder("float", shape=[None, 784])
	y = tf.placeholder("float", shape=[None, 10])

	# define graph
	feedforward = feed_forward(x)
	cost = loss(feedforward, y)
	global_step = tf.Variable(0, name='global_step', trainable=False)
	train_op = train(cost,global_step)
	evalu = evaluate(feedforward, y)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for epoch in range(epochs):

		total_batch = int(X_train.shape[0] / batch_size)


		for i in range(total_batch):
			# create batch for data
			minibatch_indices = np.random.choice(X_train.shape[0], batch_size)
			minibatch_x = X_train[minibatch_indices,]
			minibatch_y = y_train[minibatch_indices,]

			batch_fd = {x:minibatch_x, y:minibatch_y}
			minibatch_cost= sess.run(cost, feed_dict=batch_fd)
			tr = sess.run(train_op, feed_dict = batch_fd)
			#print minibatch_cost

		if epoch % display_step == 0:
			test_fd = {x:X_test, y:y_test}
			accuracy = sess.run(evalu, feed_dict=test_fd)
			print "Epoch #: {},  Test Error: {}".format(epoch, 1-accuracy)







