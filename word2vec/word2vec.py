## word2vec implementation in tensorflow
## skip-gram model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from data_preprocess_word2vec import process_data

#BATCH_SIZE = 128
#VOCAB_SIZE = 50000
#EMBED_SIZE = 124 # dimension of word embedding vector
#NUM_SAMPLED = 64 # num of negative examples to sample
#LEARNING_RATE = 0.05
#NUM_TRAIN_STEPS = 10000
SKIP_STEP = 100
#SKIP_WINDOW = 1 # context window size



class SkipGramModel:
    """Construct graph for word2vec model"""
    
    def __init__(self, vocab_size, embed_size, batch_size, learning_rate, num_sampled):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_sampled = num_sampled
        self.global_step = tf.Variable(0, dtype=tf.int32, name="global_step", trainable=False)

    def _create_placeholders(self):
        """Step 1: Defines placeholders for input and output data.
        Unlike tf.Variable, these never mutate.
        """
        with tf.name_scope("data"):
            self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name="center_words")
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size,1], name="target_words")

    def _create_embedding(self):
        """Step 2: Define weights for model.
        For word2vec, this corresponds to the weights of the hidden layer of the neural network.
        Each word will learn self.embed_size features, so it will be of size [self.vocab_size x self.embed_size]
        """
        with tf.name_scope("embedding_layer"):
            self.embed_mat = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size],-1.0, 1.0), name="embed_matrix")

    def _create_loss(self):
        """Step 3 & 4: (3) Create inference model (4) Define loss.
        For word2vec, we'll opt to use NCE for it's nice properties
        """
        with tf.name_scope("loss"):
            # define inference (forward propagation)
            embed = tf.nn.embedding_lookup(self.embed_mat, self.center_words, name="embed")
            
            # define loss
            nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size], stddev=1./(self.embed_size**0.5)), name="nce_weights")
            nce_bias = tf.Variable(tf.zeros([self.batch_size]), name="nce_bias")

            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                                    biases=nce_bias,
                                                    labels=self.target_words,
                                                    inputs=embed,
                                                    num_sampled=self.num_sampled,
                                                    num_classes=self.vocab_size), name="loss")

    def _create_optimizers(self):
        """Step 5: Define optimizer for back-propagation"""
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)


    def _create_summaries(self):
        """Define summary ops (this if for tensorboard)"""
        with tf.name_scope('summaries'):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            # merge
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """Build graph for model"""
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizers()
        self._create_summaries()

def train_model(model, batch_gen, num_train_steps):
    """Executes graph for model"""
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    initial_step = 0

    with tf.Session() as sess:
        #init variables
        sess.run(init)
        
        
        # run file for tensorboard
        writer = tf.summary.FileWriter('log_dir/word2vec', sess.graph)
        total_loss = 0.0
        initial_step = model.global_step.eval()
        for index in xrange(initial_step, initial_step + num_train_steps):
            # load in data
            center, target = batch_gen.next()
            feed_dict={model.center_words:center, model.target_words:target}
            loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)
            writer.add_summary(summary, global_step = index)
            total_loss += loss_batch
            if index % SKIP_STEP == 0:
                print('Average loss at step {}: {:.4f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, 'checkpoints/skip-gram', index)













#
## ~~~~~~~ Phase 1 ~~~~~~~~~~~~~
#def word2vec(batch_gen):
#    """ Build the graph for word2vec model and train it """
#    
#    with tf.name_scope('data'):
#        # Step 1: define the placeholders for input and output
#        center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
#        target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')
#    
#    with tf.name_scope('embedding'):
#        # Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
#        # Step 2: define weights. In word2vec, it's actually the weights that we care about
#        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), 
#                                name='embed_matrix')
#    
#        # Step 3: define the inference
#        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
#    
#    with tf.name_scope('loss'):
#        # Step 4: construct variables for NCE loss
#        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
#                                                    stddev=1.0 / (EMBED_SIZE ** 0.5)), 
#                                                    name='nce_weight')
#        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')
#    
#        # define loss function to be NCE loss function
#        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
#                                            biases=nce_bias, 
#                                            labels=target_words, 
#                                            inputs=embed, 
#                                            num_sampled=NUM_SAMPLED, 
#                                            num_classes=VOCAB_SIZE), name='loss')
#    
#    # Step 5: define optimizer
#    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
#    
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#
#        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
#        writer = tf.summary.FileWriter('log_dir/word2vec', sess.graph)
#        for index in xrange(NUM_TRAIN_STEPS):
#            centers, targets = batch_gen.next()
#            loss_batch, _ = sess.run([loss, optimizer], 
#                                    feed_dict={center_words: centers, target_words: targets})
#            total_loss += loss_batch
#            if (index + 1) % SKIP_STEP == 0:
#                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
#                total_loss = 0.0
#        writer.close()
#
def main():
    #batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    #word2vec(batch_gen)
    VOCAB_SIZE = 50000
    BATCH_SIZE = 128
    EMBED_SIZE = 128 # dimension of the word embedding vectors
    SKIP_WINDOW = 1 # the context window
    NUM_SAMPLED = 64    # Number of negative examples to sample.
    LEARNING_RATE = 1.0
    NUM_TRAIN_STEPS = 100000
    WEIGHTS_FLD = 'processed/' 
    model = SkipGramModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    train_model(model, batch_gen, NUM_TRAIN_STEPS)
    print("done")

if  __name__ == '__main__':
    main()
