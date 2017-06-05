"""
RNN Sequence Classifier
Modified from https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159
and https://gist.github.com/danijar/3f3b547ff68effb03e20c470af22c696
"""
import functools
import tensorflow as tf
import numpy as np
import os
from random import shuffle
import time

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceClassification:

    def __init__(self, data, target, dropout, lr=0.003, num_hidden=200, num_layers=3):
        self.data = data
        self.target = target
        self.dropout = dropout
	self.lr = lr
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize
    
    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        network = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.GRUCell(self._num_hidden), 
		output_keep_prob=self.dropout) for _ in range(self._num_layers)])
        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32, sequence_length=self.length)
        # Select last relevant output.
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (self.length - 1)
        flat = tf.reshape(output, [-1, output_size])
        last = tf.gather(flat, index) 
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def optimize(self):
        learning_rate = self.lr
        optimizer = tf.train.AdamOptimizer(learning_rate)
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return optimizer.minimize(cross_entropy)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        cm = tf.contrib.metrics.confusion_matrix(tf.argmax(self.target, 
		1), tf.argmax(self.prediction, 1))
	return error, cm

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.0, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

def get_feat_labels():
    targets = {} 
    classes = set()
    with open('./labels') as f:
        for line in f:
            example = "_".join(line.split("\t")[0].split("_")[:-1])
            label = line.split("\t")[1]
            targets[example] = label
	    classes.add(label)
    f.close()    
    features = {}
    m = 0
    for dirpath, dirnames, filenames in os.walk('./feat_pickles/'):
	for f in filenames:
	    example = f[:-8]
            fs = np.load(dirpath + f)
            m = max(m, fs.shape[0])
	    features[example] = fs  
    examples = list(set(targets.keys()) & set(features.keys()))
    shuffle(examples)
    X_train, y_train, X_test, y_test = [], [], [], []
    split = 3*len(examples)/4
    classes = sorted(classes)
    print "Classes:", classes
    for i, ex in enumerate(examples):
	fs = features[ex]
	fs = np.pad(fs, ((0,m-fs.shape[0]),(0,0)), 'constant', constant_values=0)
        idx = classes.index(targets[ex])
        label = np.zeros(len(classes))
        label[idx] = 1.0
        if i < split:
            X_train.append(fs)
	    y_train.append(label)
	else:
	    X_test.append(fs)
            y_test.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test

def main():
    print "Getting features..." 
    X_train, y_train, X_test, y_test = get_feat_labels() 
    num_examples, time_steps, num_features = X_train.shape
    _, num_classes = y_train.shape
    data = tf.placeholder(tf.float32, [None, time_steps, num_features])
    target = tf.placeholder(tf.float32, [None, num_classes])
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(data, target, dropout)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print "Training"
    for epoch in range(10):
	print "----------------------------"
        print "Epoch:", epoch
	start = time.time()
        for itr in range(100):
            idx = np.random.choice(np.arange(num_examples), 100, replace=False)
            X_batch = X_train[idx]
	    y_batch = y_train[idx]
            sess.run(model.optimize, {
                data: X_batch, target: y_batch, dropout: 0.5})
	train_error, cm = sess.run(model.error, {data: X_train, target: y_train, dropout: 1})
	print "Train error:", 100*train_error
        print "Confusion matrix"
	print cm
	end = time.time()
	print "Time:", end-start
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    test_error, cm = sess.run(model.error, {data: X_test, target: y_test, dropout: 1})
    print "Train error:", 100*train_error
    print "Confusion matrix"
    print cm
    print "Final test error:", test_error

if __name__ == '__main__':
    main()
