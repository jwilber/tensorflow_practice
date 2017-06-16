#### =========================================================================================
#### =========================================================================================
###### Convolutional Neural Network Example with tensorflow
######  		---- Trained on MNIST ----
#### =========================================================================================
#### =========================================================================================

# import stuff we'll need
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
## Part 3: Define Helper Functions to Construct Conv Net
#### =========================================================================================

def layer(input_, weight_shape, bias_shape):
	"""Creates full-connected layer for neural network"""
	weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
	bias_init = tf.constant_initializer(value=0)
	W = tf.get_variable("W", shape=weight_shape, initializer=weight_init)
	b = tf.get_variable("bias", shape=bias_shape, initializer=bias_init)
	return tf.nn.relu( tf.matmul(input_,W) + b)

def conv2d(input_, weight_shape, bias_shape):
	"""Creates convolutional layer"""
	inp = weight_shape[0] * weight_shape[1] * weight_shape[2]
	# define weights
	weight_init = tf.random_normal_initializer(stddev = (2.0 /inp)**0.5 )
	W = tf.get_variable("W", shape=weight_shape, initializer=weight_init)
	# define bias
	bias_init = tf.constant_initializer(value=0)
	b = tf.get_variable("bias", shape=bias_shape, initializer=bias_init)
	conv_output = tf.nn.conv2d(input_, W, strides=[1,1,1,1], padding='SAME')
	return tf.nn.relu( conv_output)

def max_pool(input_, k=2):
	"""Generates max-pooling layer with non-overlapping window of size k"""
	return tf.nn.max_pool(input_, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')


def inference(x):
	"""Forward pass of convolutional neural network"""
	# reshape mnist to image size
	x = tf.reshape(x,[-1, 28, 28,1])

	with tf.variable_scope('conv_1'):
		conv1 = conv2d(x, [5, 5, 1, 32], [32]) # filter_width, filter_height, input_filters, num_filters
		out1 = max_pool(conv1, k=2)

	with tf.variable_scope('conv_2'):
		conv2 = conv2d(out1, [5, 5, 32, 64], [64])
		out2 = max_pool(conv2, k=2)

	with tf.variable_scope('fc'):
		out2 = tf.reshape(out2,[-1, 7*7*64])
		fc = layer(out2, [7*7*64, 1024], [1024])
		# apply dropout
		out3 = tf.nn.dropout(fc, keep_prob)

	with tf.variable_scope('output'):
		out4 = layer(out3, [1024,10], [10])
		return out4

def loss(output, y):
	"""Calculates cross-entropy loss"""
	xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
	return tf.reduce_mean(xentropy)

def evaluate(output, y):
	"""Calculates accuracy for model"""
	# prediction for output
	out_pred = tf.argmax(output,1)
	y_pred = tf.argmax(y, 1)
	return tf.reduce_mean(tf.cast(tf.equal(out_pred,y_pred), tf.float32))

def train(cost, global_step):
	"""Training regiment for cnn"""
	tf.summary.scalar('cost', cost)
	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(cost, global_step=global_step)
	return train_op


#### =========================================================================================
## Part 4: Set up graph
#### =========================================================================================

batch_size = 32
epochs = 100
total_batch = int(1.0*X_train.shape[0] / batch_size)
display_step = 1
keep_prob = 0.5


with tf.Graph().as_default():

	# init placeholders for inputs
	x = tf.placeholder("float", shape=[None, 784])
	y = tf.placeholder("float", shape=[None,10])

	##### construct graph
	# first, feed forward
	feed_forward = inference(x)
	# then, calculate loss
	cost = loss(feed_forward, y)
	# next, backpropagation
	global_step = tf.Variable(0, name='global_step', trainable=False)
	backprop = train(cost, global_step)
	# finally, evaluate model
	performance = evaluate(feed_forward, y)


	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for epoch in range(epochs):

		for batch in range(total_batch):

			# create batch for data
			minibatch_indices = np.random.choice(X_train.shape[0], batch_size)
			minibatch_x = X_train[minibatch_indices,]
			minibatch_y = y_train[minibatch_indices,]

			# feed forward, train, and get cost
			f_dict = {x:minibatch_x, y:minibatch_y}
			sess.run(backprop, feed_dict=f_dict)

		if epoch % display_step == 0:
			test_fd = {x:X_test, y:y_test}
			acc = sess.run(performance, feed_dict=test_fd)
			print "Epoch #: {}, Loss: {}".format(epoch, 1-acc)








