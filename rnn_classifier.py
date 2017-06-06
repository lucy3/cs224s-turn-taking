"""
RNN Sequence Classifier

Model modified from:
    https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159
    and https://gist.github.com/danijar/3f3b547ff68effb03e20c470af22c696
Tensorboard embedding using:
    https://stackoverflow.com/questions/41258391/tensorboard-embedding-example
"""
import functools
import tensorflow as tf
import numpy as np
import os
from random import shuffle
import time
from collections import Counter
from tensorflow.contrib.tensorboard.plugins import projector
import sys

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

    def __init__(self, data, target, addition, dropout, lr=0.002, num_hidden=3, num_layers=1):
        self.data = data
        self.target = target
        self.addition = addition
        self.dropout = dropout
        self.lr = lr
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

        print("Num layers", num_layers)
        print("Num hidden", num_hidden)
        print("Learning rate", lr)
    
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
        augmented = tf.concat([last, self.addition], 1)
        weight, bias = self._weight_and_bias(
            self._num_hidden + int(self.addition.get_shape()[1]), int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(augmented, weight) + bias)
        return prediction, last

    @lazy_property
    def optimize(self):
        learning_rate = self.lr
        optimizer = tf.train.AdamOptimizer(learning_rate)
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction[0]))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        return optimizer.minimize(cross_entropy + sum(reg_losses)*0.01)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction[0], 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        cm = tf.contrib.metrics.confusion_matrix(tf.argmax(self.target, 
    	    1), tf.argmax(self.prediction[0], 1))
        return error, cm

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.0, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

def write_metadata(used_exs, idx, lexical, targets):
    with open('./log/metadata.tsv', 'w') as f:
        these_idx = idx[int(4*len(used_exs)/6):int(5*len(used_exs)/6)]
        f.write('word\tlabel\n')
        for i in range(len(these_idx)):
            this_i = these_idx[i]
            ex = used_exs[this_i]
            f.write(lexical[ex] + '\t' + targets[ex] + '\n')

def get_feat_labels():
    targets = {} 
    lexical = {}
    vocab = set()
    classes = []
    with open('./labels') as f:
        for line in f:
            contents = line.split("\t")[0].split("_")
            example = "_".join(contents[:-1])
            lex = contents[-1]
            vocab.add(lex)
            lexical[example] = lex
            label = line.split("\t")[1]
            targets[example] = label
            classes.append(label)
    least = Counter(classes).most_common()[-1][1]
    classes = set(classes)
    f.close()
    features = {}
    m = 0
    for dirpath, dirnames, filenames in os.walk('./feat_pickles/'):
        for f in filenames:
            example = f[:-8]
            if example in targets.keys():
                fs = np.load(dirpath + f)
                m = max(m, fs.shape[0])
                features[example] = fs  
    examples = list(set(targets.keys()) & set(features.keys()))
    shuffle(examples)
    X, y, z = [], [], []
    split1 = 5*len(examples)/6
    split2 = 4*len(examples)/6
    classes = sorted(classes)
    vocab = sorted(vocab)
    print("Classes:", classes)
    print("Vocab:", vocab)
    c = Counter()
    used_exs = []
    for i, ex in enumerate(examples):
        if c[targets[ex]] > least:
            continue
        fs = features[ex]
        fs = np.pad(fs, ((0,m-fs.shape[0]),(0,0)), 'constant', constant_values=0)
        c[targets[ex]] += 1
        idx = classes.index(targets[ex])
        label = np.zeros(len(classes))
        label[idx] = 1.0
        lex = np.zeros(len(vocab))
        idxl = vocab.index(lexical[ex])
        lex[idxl] = 1.0
        used_exs.append(ex)
        X.append(fs)
        y.append(label)
        z.append(lex)
    # seperate out our training, val, test sets
    idx = list(range(len(X)))
    shuffle(idx)
    train_idx = idx[:int(4*len(X)/6)]
    val_idx = idx[int(4*len(X)/6):int(5*len(X)/6)]
    test_idx = idx[int(5*len(X)/6):]
    write_metadata(used_exs, idx, lexical, targets)
    X = np.array(X)
    y = np.array(y)
    z = np.array(z)
    X_train = X[train_idx]
    y_train = y[train_idx]
    z_train = z[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    z_val = z[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    z_test = z[test_idx]
    return X_train, y_train, z_train, X_val, y_val, z_val, X_test, y_test, z_test

def main():
    print("Getting features...")
    X_train, y_train, z_train, X_val, y_val, z_val, X_test, y_test, z_test = get_feat_labels() 
    num_examples, time_steps, num_features = X_train.shape
    print("# of training examples:", num_examples)
    _, num_classes = y_train.shape
    _, num_vocab = z_train.shape
    data = tf.placeholder(tf.float32, [None, time_steps, num_features])
    target = tf.placeholder(tf.float32, [None, num_classes])
    addition = tf.placeholder(tf.float32, [None, num_vocab])
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(data, target, addition, dropout)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print("Training")
    with sess.as_default():
        for epoch in range(500):
            start = time.time()
            l = list(range(num_examples))
            shuffle(l)
            idx_splits = [l[i:i + 200] for i in range(0, len(l), 200)]
            for itr in range(len(idx_splits)):
                idx = idx_splits[itr] 
                X_batch = X_train[idx]
                y_batch = y_train[idx]
                z_batch = z_train[idx]
                sess.run(model.optimize, {
                    data: X_batch, target: y_batch, addition: z_batch, dropout: 0.2})
            if epoch % 50 == 0:
                print("----------------------------")
                print("Epoch:", epoch)
                print("Calculating error on 100 training examples...")
                idx = np.random.choice(np.arange(num_examples), 100, replace=False)
                train_error, cm = sess.run(model.error, {data: X_train[idx], 
    			      target: y_train[idx], addition: z_train[idx], dropout: 1})
                print("Train error:", 100*train_error)
                print("Confusion matrix")
                print(cm)
                print("Calculating error on validation set...")
                val_error, cm = sess.run(model.error, {data: X_val, target: y_val, addition: z_val, dropout: 1})
                print("Val error:", 100*val_error)
                print("Confusion matrix")
                print(cm)
                end = time.time()
                print("Time:", end-start)
            if epoch == 499:
                _, embedding = sess.run(model.prediction, {data: X_val, target: y_val, addition: z_val, dropout: 1})
                X = tf.Variable([0.0], name='embedding')
                place = tf.placeholder(tf.float32, shape=embedding.shape)
                set_x = tf.assign(X, place, validate_shape=False)
                sess.run(set_x, feed_dict={place: embedding})
                summary_writer = tf.summary.FileWriter('log', sess.graph)
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = 'embedding:0'
                embedding_conf.metadata_path = os.path.join('log', 'metadata.tsv')
                projector.visualize_embeddings(summary_writer, config)
                saver = tf.train.Saver()
                saver.save(sess, os.path.join('log', "model.ckpt"))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    test_error, cm = sess.run(model.error, {data: X_test, target: y_test, addition: z_test, dropout: 1})
    print("Test error:", 100*test_error)
    print("Confusion matrix")
    print(cm)

if __name__ == '__main__':
    main()
