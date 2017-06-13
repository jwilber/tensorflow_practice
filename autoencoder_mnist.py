#### =========================================================================================
#### =========================================================================================
###### Autoencoder Neural Network Example with tensorflow
######  		---- Trained on MNIST ----
#### =========================================================================================
#### =========================================================================================

# import stuff we'll need
import argparse
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
def mnist_onehot(label):
	"""one-hot encodes label"""
	a = label 
	b = np.zeros((label.shape[0], 10))
	b[np.arange(label.shape[0]), a] = 1
	return b 

y_train = mnist_onehot(y_train)
y_test = mnist_onehot(y_test)


# print "X_train shape: {}".format(X_train.shape)
# print "X_test shape: {}".format(X_test.shape)
# print "y_train shape:{}".format(y_train.shape)
# print "y_test shape:{}".format(y_test.shape)



#### =========================================================================================
## Part 3: Define Helper Functions to Construct Autoencoder
#### =========================================================================================

def layer(input_, weight_shape, bias_shape):
	"""Create layer for feed-forward network"""

	weight_init = tf.random_normal_initializer(stddev = (2.0 / weight_shape[0])**0.5)
	bias_init = tf.constant_initializer(value=0)
	W = tf.get_variable(name="W", shape=weight_shape, initializer=weight_init)
	b = tf.get_variable(name='bias', shape=bias_shape, initializer=bias_init)
	logits = tf.matmul(input_, W) + b 
	return tf.nn.sigmoid(logits)

# An Autoencoder is just a decoder, follower by an encoder, so we'll define the two separately

def encoder(x, des_shape):
	"""Encodes x down to desired shape"""

	# 3 layers, so get shapes
	shape_1 = 1000.
	shape_2 = 500.
	shape_3 = 250.

	with tf.variable_scope('decoder'):

		with tf.variable_scope('decode_1'):
			shrink_1 = layer(x, [784, shape_1], [shape_1])
			print "shrink_1 shape:{}".format(shrink_1.shape)

		with tf.variable_scope('decode_2'):
			shrink_2 = layer(shrink_1, [shape_1, shape_2], [shape_2])
			print "shrink_2 shape:{}".format(shrink_2.shape)

		with tf.variable_scope('decode_3'):
			shrink_3 = layer(shrink_2, [shape_2, shape_3], [shape_3])
			print "shrink_3 shape:{}".format(shrink_3.shape)

		with tf.variable_scope('decode_4'):
			shrink_4 = layer(shrink_3, [shape_3, des_shape], [des_shape])
			print "shrink_4 shape:{}".format(shrink_4.shape)

	return shrink_4	

def decoder(x, des_shape):
	"""dencodes x back to size: fin_shape"""

	# Shoot shape back to 784 (fin_shape)
	shape_1 = 250.
	shape_2 = 500.
	shape_3 = 1000.

	with tf.variable_scope('encoder'):

		with tf.variable_scope('encode_1'):
			shrink_1 = layer(x, [2., shape_1], [shape_1])
			print "shrink_1 shape:{}".format(shrink_1.shape)

		with tf.variable_scope('encode_2'):
			shrink_2 = layer(shrink_1, [shape_1, shape_2], [shape_2])
			print "shrink_2 shape:{}".format(shrink_2.shape)

		with tf.variable_scope('encode_3'):
			shrink_3 = layer(shrink_2, [shape_2, shape_3], [shape_3])
			print "shrink_3 shape:{}".format(shrink_3.shape)

		with tf.variable_scope('encode_4'):
			shrink_4 = layer(shrink_3, [shape_3, des_shape], [des_shape])
			print "shrink_4 shape:{}".format(shrink_4.shape)

	return shrink_4

def loss(original, reconstruction):
	"""Calculates L2-loss between original image and reconstructed image"""

	with tf.variable_scope('training'):
		# calculate l2 loss across all rows
		l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(original, reconstruction)),1))
		# take mean of this 
		train_loss = tf.reduce_mean(l2)
		train_summary_op = tf.summary.scalar('loss_op', train_loss)
		return train_loss, train_summary_op

def training(cost, global_step):
	"""Train autoencoder by minimizing loss"""
	optimizer = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9, beta2 = 0.999, epsilon=1e-8,
		use_locking=False, name='Adam')
	train_op = optimizer.minimize(cost,global_step=global_step)
	return train_op

def image_summary(summary_label, tensor):
	"""Reshape input tensor and return its summary"""

	tensor_reshaped = tf.reshape(tensor, [-1, 28, 28, 1])
	return tf.summary.image(summary_label, tensor_reshaped)


def evaluate(reconstructed, original):
 	"""Compare reconstructed and test image"""

 	with tf.variable_scope('validation'):
 		in_im_op = image_summary('input_image', original)
 		out_im_op = image_summary('output_image', reconstructed)
 		# calculate loss
 		l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(reconstructed, original)), 1))
 		val_loss = tf.reduce_mean(l2)
 		val_summary_op = tf.summary.scalar('val_cost', val_loss)
 		return val_loss, in_im_op, out_im_op, val_summary_op


#### =========================================================================================
## Part 4: Set up and run graph
#### =========================================================================================

# # test
# batch_size = 32

# with tf.Graph().as_default():

# 	x = tf.placeholder('float', shape=[None, 784])

# 	decode = decoder(x, 2.)
# 	autoencoder = encoder(decode, 784.)
# 	cost = loss(x, autoencoder)



# 	sess = tf.Session()
# 	sess.run(tf.global_variables_initializer())

# 	minibatch_indices = np.random.choice(X_train.shape[0], batch_size)
# 	minibatch_x = X_train[minibatch_indices,]
# 	minibatch_y = y_train[minibatch_indices,]

# 	#sess.run(autoencoder, feed_dict = {x:minibatch_x})
# 	sess.run(cost, feed_dict = {x:minibatch_x})


epochs = 100
batch_size = 32

if __name__ == '__main__':

	# parser = argparse.ArgumentParser(description='Test various optimization strats')
	# parser.add_argument('n_code', nargs=1, type=str)
	# args = parser.parse_args()
	# n_code = args.n_code[0]


	with tf.Graph().as_default():

		with tf.variable_scope('autoencoder_model'):

			x = tf.placeholder("float", shape=[None,784])

			code = encoder(x, 2.)
			output = decoder(code, 784.)
			cost, train_summary_op = loss(output, x)
			global_step = tf.Variable(0, name='global_step', trainable=False)
			print cost
			train_op = training(cost, global_step=global_step)
			eval_op, in_im_op, out_im_op, val_summary_op = evaluate(output, x)
			#summary_op = tf.summary.merge()
			#saver = tf.train.Saver(max_to_keep=200)

			sess = tf.Session()
			sess.run(tf.global_variables_initializer())

			# training cycle
			for epoch in range(epochs):

				avg_cost = 0.
				total_batch = int(1.*X_train.shape[0] / batch_size)
				minibatch_indices = np.random.choice(X_train.shape[0], batch_size)
				minibatch_x = X_train[minibatch_indices,]
				minibatch_y = y_train[minibatch_indices,]

				loss, trainn = sess.run([cost, train_op], feed_dict = {x:minibatch_x})
				print loss








