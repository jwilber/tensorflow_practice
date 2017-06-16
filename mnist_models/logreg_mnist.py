# ================================================================================
# ================================================================================
# ================================================================================

# Logistic Regression Using Tensorflow
# The data was loaded from mnist to simulate loading of real data

# ================================================================================
# ================================================================================
# ================================================================================


import tensorflow as tf 
import numpy as np
from sklearn.model_selection import train_test_split


# ================================================================================
# Load data
# ================================================================================
from keras.datasets import mnist
img_rows, img_cols = 28, 28
#the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32')

# Split X_test into test set and validation set
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.4, random_state=42)
X_train /= 255
X_test /= 255
X_val /= 255

print "X_test shape:", X_test.shape
print "X_train shape:", X_train.shape
print "X_val shape:", X_val.shape
print "y shapes:", y_test.shape, y_train.shape, y_val.shape

# still need to one-hot encode y values
def one_hot_encode(x, n_classes): return np.eye(n_classes)[x]

y_test = one_hot_encode(y_test, 10)
y_train = one_hot_encode(y_train, 10)
y_val = one_hot_encode(y_val, 10)
print "y shapes:", y_test.shape, y_train.shape, y_val.shape


# ================================================================================
# Set up graph
# ================================================================================

# constant parameters
n_epochs = 5
batch_size = 10
lr = 0.01

# set variables/placeholders
x = tf.placeholder(tf.float32, shape=[batch_size, 784], name='image')
y = tf.placeholder(tf.float32, shape=[batch_size, 10], name='label')

w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros(shape=[1, 10], name='bias'))

# define inference
logits = tf.matmul(x,w) + b

# define loss
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)


# ================================================================================
# Exectute Graph in Session
# ================================================================================

init = tf.global_variables_initializer()

with tf.Session() as sess:

	# set up writer for tensorboard use
	writer = tf.summary.FileWriter('log_dir', sess.graph)

	# init vars
	sess.run(init)

	# set up batches
	n_batches = int(X_train.shape[0] / batch_size)
	total_correct_preds = 0.

	for i in  range(n_epochs):

		for _ in range(n_batches):

			train_batch_index = np.random.choice(range(X_train.shape[0]), batch_size)
			X_batch = X_train[train_batch_index,:]
			y_batch = y_train[train_batch_index]
			print "Batch shapes - X: {}, y: {}".format(X_batch.shape, y_batch.shape)

			test_batch_index = np.random.choice(range(X_test.shape[0]), batch_size)
			X_test_batch = X_test[test_batch_index,:]
			y_test_batch = y_test[test_batch_index]
			print "Batch shapes - X: {}, y: {}".format(X_test_batch.shape, y_test_batch.shape)

			_, cost_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict = {x:X_batch, y:y_batch})

			# Test stuff
			preds = tf.nn.softmax(logits_batch)
			correct_pred = tf.equal(tf.argmax(preds,1), tf.argmax(y_batch,1))
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			total_correct_preds += sess.run(accuracy, feed_dict = {x: X_test_batch, y:y_test_batch})
			
		print "Epoch: {}, Loss: {}, Accuracy: {:.2f}".format(i, cost_batch, 1.*total_correct_preds/X_test.shape[0])






