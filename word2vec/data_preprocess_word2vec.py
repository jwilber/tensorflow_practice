# Script to preprocess wikipedia data
# Taken from Chip Huyen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import random
import os
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
EXPECTED_BYTES = 31344016
DATA_FOLDER = '/Users/jared/Desktop/tensorflow_practice/'
FILE_NAME = 'text8.zip'
NUM_WORDS_IN_TB = 10000


def download(file_name, expected_bytes):
    """Downloads text8 dataset (if it's not already downloaded)"""
    file_path = DATA_FOLDER + file_name
    if os.path.exists(file_path):
        #print("Dataset Ready for Use")
        return file_path

def read_data(file_path):
    """Reads data into list of tokens.
    ~ 17,005,207 tokens.
    """
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return words

def build_vocab(words, vocab_size):
    """Builds vocabulary of size vocab_size-most frequent words
    In this case, we do top 1000 words.
    Returns 2 dictionaries: 
        dict1: dictionary of top words: (word:index)
        dict2: dictionary of top words: (index:word)"""
    dictionary = dict()
    count = [('UNK', -1)]
    count.extend(Counter(words).most_common(vocab_size-1))
    index = 0
    with open('processed/vocab_{}.tsv'.format(NUM_WORDS_IN_TB), "w") as f:
        for word, _ in count:
            dictionary[word] = index
            if index < NUM_WORDS_IN_TB:
                f.write(word + "\n")
            index += 1
    index_dict = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dict

def convert_words_to_index(words, dictionary):
    """Converts words to their index values.
    Returns list, where each element is it's corresponding index
    in `dictionary`"""
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words, context_window_size):
    """Form training pairs according to skip-gram model."""
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get random target before center word
        for target in index_words[max(0, index-context): index]:
            yield center, target
        for target in index_words[index + 1: index + context + 1]:
            yield center, target

def get_batch(iterator, batch_size):
    """Group numerical stream into batches and yield them as np.arrays"""
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch

def process_data(vocab_size, batch_size, skip_window):
    """Combines all of the above functions:
    Returns batches of processed data, as generator.
    """
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    words = read_data(file_path)
    dictionary, _ = build_vocab(words, vocab_size)
    index_words = convert_words_to_index(words, dictionary)
    del words # save memory
    single_gen = generate_sample(index_words, skip_window)
    return get_batch(single_gen, batch_size)

def get_index_vocab(vocab_size):
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    words = read_data(file_path)
    return build_vocab(words, vocab_size)

#def main():
    #download(FILE_NAME, EXPECTED_BYTES)
    #file_path = DATA_FOLDER + FILE_NAME
    #words = read_data(file_path)
    #test1, test2 = build_vocab(words, 1500)    
    #print("Dictionary 1 Length:", len(test1.items()))
    #print("Dictionary items, first 10:", test1.items()[0:10])
    #print(test1["limited"])
    #print("Dictionary 2 Length:", len(test2.items()))
    #print("Dictionary 2 items, first 10:", test2.items()[0:10])
    #print
    #print("Checking convert_words_to_index")
    #print(test2[752])
    #index_words = convert_words_to_index(words, test1)
    ##print(index_words)
    #for i in range(3): print( next(process_data(100, 10, 4)) )



#if __name__ == '__main__':
#    main()
#    print("finished")
