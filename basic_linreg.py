import tensorflow as tf 
import numpy as np 


# Create data. want model to learn y = 2*x - 15
X = np.linspace(-1, 1, 100)
Y = 250.*X -23. + np.random.randn(X.shape[0])


# Create graph
x = tf.placeholder(dtype='float32')
y = tf.placeholder(dtype='float32')

w = tf.Variable(tf.zeros_like(X, dtype='float32'), name='weight', dtype='float32')
b = tf.Variable(tf.zeros_like(X, dtype='float32'), name='bias', dtype='float32')



y_preds = w*x + b 

loss = tf.reduce_sum(tf.square(y - y_preds))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# construct graph
with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	for i in range(1000):

		for j in range(X.shape[0]):
			x_t = X[j]
			y_t = Y[j]

			w_,b_,loss_, _ = sess.run([w,b,loss,optimizer], feed_dict = {x:x_t, y:y_t})

		w_, b_ = sess.run([w,b], feed_dict={x:x_t, y:y_t})

		if i % 5 == 0: 
			print "weight: {}, bias: {}, total loss: {}".format(np.mean(w_), np.mean(b_), loss_)