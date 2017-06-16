#### =========================================================================================
#### =========================================================================================
###### Logistic Regression Example with tensorflow
######
#### =========================================================================================
#### =========================================================================================

import tensorflow as tf
from keras.datasets import mnist
import numpy as np 



#### =========================================================================================
##  Part 1: Define graph architecture
#### =========================================================================================


##  1. Inference: Produce probability dsn over outcome classes given a minibatch
def inference(x):
	"""Calculates P(Y=i|X) for x.
	returns: softmax output
	"""
	weight_init = tf.random_uniform_initializer(minval=-1, maxval=1)
	bias_init = tf.constant_initializer(value=0)

	tf.constant_initializer(value=0)
	W = tf.get_variable("W", shape=[784,10],initializer=weight_init)
	b = tf.get_variable("bias", shape=[10], initializer=bias_init)
	output = tf.nn.softmax( tf.matmul(x,W) + b )
	return output 

##  2. Calculate loss using cross-entropy
def loss(output, y):
	"""Calculates cross-entropy loss over a minibatch"""
	dot_prod = y * tf.log(output)  # y * log(y_hat)
	xentropy = - tf.reduce_sum(dot_prod, reduction_indices=1) # each row only has 1 non-zero value, so this works
	loss = tf.reduce_mean(xentropy)
	return loss

##  3. Training
def training(cost, global_step):
	tf.summary.scalar("cost", cost) # log cost of each minibatch
	"""Updates weights via SGD"""
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(cost, global_step=global_step)
	return train_op

##  4. Assess Model Accuracy
def evaluate(output, y):
	"""Returns Model Accuracy"""
	# Find number of equal predictions
	correct_preds = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
	# Get mean of correct (Booleans cast to float)
	return tf.reduce_mean(tf.cast(correct_preds, tf.float32))



#### =========================================================================================
## Part 2: Load in Data
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
## Part 3: Set Constants
#### =========================================================================================

# set constants
learning_rate = 0.01
epochs = 100
batch_size = 32
display_step =10
img_rows = 28
img_cols = 28


#### =========================================================================================
## Part 4: Train and log Logistic Regression Model
#### =========================================================================================

with tf.Graph().as_default():

	# first, define placeholders
	x = tf.placeholder("float", [None,784])
	y = tf.placeholder("float", [None,10])

	# next, define graph
	output = inference(x)
	cost = loss(output_, y)
	global_step = tf.Variable(0, name='global_step', trainable=False)
	train_op = training(cost, global_step)
	eval_op = evaluate(output, y)

	# Initialize Session and Variables
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())


	# Train Logistic Regression Model

	for epoch in range(epochs):

		avg_cost = 0
		total_batch = int( X_train.shape[0] / batch_size )

		# loop over all the batches
		for i in range(total_batch):

			# split data into batches
			minibatch_inidices = np.random.choice(X_train.shape[0], batch_size)
			minibatch_x = X_train[minibatch_inidices,]
			minibatch_y = y_train[minibatch_inidices]

			# feed batches into train and cost
			feed_dict = {x:minibatch_x, y:minibatch_y}
			minibatch_cost, _ = sess.run([cost, train_op], feed_dict=feed_dict)
			avg_cost += 1.0*minibatch_cost / total_batch

		# For each epoch, display logs
		if epoch % display_step == 0:
			val_feed_dict = {
			x:X_test,
			y:y_test
			}
			accuarcy = sess.run(eval_op, feed_dict = val_feed_dict)
			print "Validation Error: {}".format(1-accuarcy)




