# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import constants
import numpy as np
from sklearn.base import BaseEstimator
# from keras.layers import Convolution1D


class AttentionBasedRNN(BaseEstimator):

    def __init__(self, n_vocab, **param_map):

        for k in param_map:
            setattr(self, k, param_map[k])

        self.n_vocab = n_vocab

        # hyper parameters. values will be injected on estimator.set_params()
        self.w_decay_factor = None
        self.attn_score_func = None
        self.rnn_dim = None
        self.batch_size = None
        self.pool_len = None
        self.epoch = 3 # constant while cv
        self.lr = None
        self.n_filter = None
        self.ent_vec_dim = None
        self.attr_vec_dim = None
        self.filter_len = None
        self.cell_clip = None
        self.rnn_unit = None

        self.use_attention = True
        self.use_convolution = True

    def get_params(self, deep=False):
        return {
            'cell_clip': self.cell_clip,
            'rnn_unit': self.rnn_unit,
            'attn_score_func': self.attn_score_func,
            'lr': self.lr,
            'w_decay_factor': self.w_decay_factor,
            'rnn_dim': self.rnn_dim,
            'batch_size': self.batch_size,
            'n_filter': self.n_filter,
            'ent_vec_dim': self.ent_vec_dim,
            'attr_vec_dim': self.attr_vec_dim,
            'pool_len': self.pool_len,
            'use_convolution': [True, False],
            'filter_len': self.filter_len}

    # call build_graph() beforehand
    def fit(self, sess, ids, entity, attr, sent_lens, y, token_lookup, ent_lookup, attr_lookup):
        self.vec_dim = token_lookup.shape[1]

        sess.run(self.embedding_w_init_op, feed_dict={self.embedding_w_ph: token_lookup})
        sess.run(self.embedding_e_init_op, feed_dict={self.embedding_e_ph: ent_lookup})
        sess.run(self.embedding_a_init_op, feed_dict={self.embedding_a_ph: attr_lookup})

        n_batch = len(ids) // self.batch_size
        total = self.batch_size * n_batch

        for e in range(self.epoch):
            perm = np.random.permutation(total)
            for step, i in enumerate(range(0, total, self.batch_size)):
                batch = self._make_batch(perm[i: i + self.batch_size], ids, entity, attr, sent_lens, y)
                sess.run(self.train_op, feed_dict=batch)
                if step % 32 == 0 or step == n_batch - 1:
                    acc, loss = sess.run([self.accuracy_op, self.loss], feed_dict=batch)
                    print("batch {}/{}  acc: {:.5f} loss: {:.5f}".format(step+1, n_batch, acc, loss))

            print 'Finish epoch {}/{}: trained with {} sentences'.format(e+1, self.epoch, len(ids))

    def _make_batch(self, indices, ids, entity, attr, sent_lens, labels):
        return {self.ids_ph: ids[indices, :],
                self.entity_ph: np.expand_dims(entity[indices], 1),
                self.attr_ph: np.expand_dims(attr[indices], 1),
                self.y_ph: labels[indices],
                self.sent_len_ph: sent_lens[indices]/self.pool_len,
                self.b_size_ph: self.batch_size,
                self.pool_len_ph: self.pool_len,
                }

    def eval(self, sess, ids, entity, attr, sent_lens, labels):
        feed_dict = \
            {self.ids_ph: ids,
             self.entity_ph: np.expand_dims(entity, 1),
             self.attr_ph: np.expand_dims(attr, 1),
             self.sent_len_ph: sent_lens/self.pool_len,
             self.pool_len_ph: self.pool_len,
             self.b_size_ph: len(ids),
             self.y_ph: labels}

        return self.accuracy_op.eval(feed_dict, sess)

    def build_graph(self, n_ent, n_attr, vec_dim, label_num):

        def conv1d(x, W, b):
            """
            :param x: (?, sent, vec)
            :param W: (row, col=vec, in_ch=1, out_ch)
            :param b: (?, sent, n_filter)
            :return:
            """
            padding = tf.zeros((tf.shape(x)[0], self.filter_len-1, tf.shape(x)[2]), dtype=tf.float32)
            # x = (batch, sent, vec_dim)
            x = tf.concat(1, (padding, x))
            x = tf.expand_dims(x, 3)
            x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
            x = tf.squeeze(x, [2])
            x = tf.nn.bias_add(x, b)
            x = tf.nn.relu(x)
            return x

        def hard_sigmoid(x):
            x = (0.2 * x) + 0.5
            zero = tf.convert_to_tensor(0., tf.float32)
            one = tf.convert_to_tensor(1., tf.float32)
            return tf.clip_by_value(x, zero, one)

        def repeat(x, times, axis):
            x_shape = x.get_shape().as_list()
            # slices along the repeat axis
            splits = tf.split(axis, x_shape[axis], x)
            # repeat each slice the given number of reps
            x_rep = [s for s in splits for _ in range(times)]
            return tf.concat(axis, x_rep)

        def attn_score(string):
            if string == 'sigmoid':
                def a(x):
                    y = tf.nn.sigmoid(x)
                    return y/tf.reduce_sum(y, axis=0)
                return a
            elif string == 'h_sigmoid':
                def a(x):
                    y = hard_sigmoid(x)
                    return y/tf.reduce_sum(y, axis=0)
                return a
            elif string == 'softmax':
                return tf.nn.softmax

        def fo_pool(x):
            """
            :param x: (?, sent, vec*3)
            :return:  (?, sent, vec)
            """
            # all tensors are (batch_size, nb_filter/3)
            x_shape = tf.shape(x)
            self.cell = tf.zeros((x_shape[0], x_shape[2]/3), dtype=tf.float32)
            c_f, c_i, c_o = tf.split(2, 3, x)

            # process word
            def step(idx):
                f = tf.nn.sigmoid(c_f[:, idx, :])
                i = tf.nn.tanh(c_i[:, idx, :])
                o = tf.nn.sigmoid(c_o[:, idx, :])

                self.cell = tf.mul(self.cell, f) + tf.mul(i, 1 - f)
                h = tf.mul(self.cell, o)
                return h

            # (sent, batch, vec)
            hs_ = tf.map_fn(step, tf.range(0, x_shape[1]), dtype=tf.float32)
            return tf.transpose(hs_, [1, 0, 2])

        def xavier_init(shape, name):
            return tf.get_variable(name, shape=shape, initializer=xavier_initializer(), dtype=tf.float32)

        # Define operations and tensors in `graph`.
        # ------------
        # placeholders
        # ------------
        self.b_size_ph = tf.placeholder(shape=(), dtype=tf.int32)
        self.pool_len_ph = tf.placeholder(shape=(), dtype=tf.int32)
        self.embedding_w_ph = tf.placeholder(shape=(self.n_vocab, vec_dim), dtype=tf.float32)
        self.embedding_e_ph = tf.placeholder(shape=(n_ent, vec_dim), dtype=tf.float32)
        self.embedding_a_ph = tf.placeholder(shape=(n_attr, vec_dim), dtype=tf.float32)
        self.ids_ph = tf.placeholder(shape=(None, constants.max_sent_len,), dtype=tf.int32)
        self.entity_ph = tf.placeholder(shape=(None, 1,), dtype=tf.int32)
        self.attr_ph = tf.placeholder(shape=(None, 1,), dtype=tf.int32)
        self.y_ph = tf.placeholder(shape=(None, label_num), dtype=tf.int32)
        self.sent_len_ph = tf.placeholder(tf.int32, shape=(None,))

        # ---------
        # variables
        # ---------
        # for embedding (values will be override by `assign`)
        self.embed_tok = tf.Variable(tf.random_uniform((self.n_vocab, vec_dim)))
        self.embed_ent = tf.Variable(tf.random_uniform((n_ent, vec_dim)))
        self.embed_attr = tf.Variable(tf.random_uniform((n_attr, vec_dim)))
        # for aspect projection
        # Went = xavier_init((vec_dim, self.ent_vec_dim), name='Went')
        # Wattr = xavier_init((vec_dim, self.attr_vec_dim), name='Wattr')
        # bv = tf.Variable(tf.zeros((self.rnn_dim,)))
        # for attention
        Wa_ent = xavier_init((self.n_filter + self.ent_vec_dim, 1), name='Wa_ent')
        Wa_attr = xavier_init((self.n_filter + self.attr_vec_dim, 1), name='Wa_attr')
        # for conv (row, col=vec, in_ch, out_ch)
        Wc = xavier_init((self.filter_len, vec_dim, 1, self.n_filter*3), name='conv')
        bc = tf.Variable(tf.zeros((self.n_filter*3,)), name='convb')
        Wc1 = xavier_init((self.filter_len, self.n_filter, 1, self.n_filter*3), name='conv1')
        bc1 = tf.Variable(tf.zeros((self.n_filter*3,)), name='convb1')
        # before prediction
        Wx = xavier_init((self.n_filter, self.n_filter), name='Wx')
        Wp = xavier_init((self.n_filter, self.n_filter), name='Wp')
        # for prediction
        Ws = xavier_init((self.n_filter, label_num), name='Ws')

        # ----
        # flow
        # ----
        self.embedding_w_init_op = tf.assign(self.embed_tok, self.embedding_w_ph)
        self.embedding_e_init_op = tf.assign(self.embed_ent, self.embedding_e_ph)
        self.embedding_a_init_op = tf.assign(self.embed_attr, self.embedding_a_ph)

        # (?, 60, vec)
        words = tf.gather(self.embed_tok, self.ids_ph)
        # (?, 60, n_filter*3)
        words = conv1d(words, Wc, bc)
        # (?, 60, n_filter)
        words = fo_pool(words)
        # words = conv1d(words, Wc1, bc1)
        # words = fo_pool(words)

        ent = tf.squeeze(tf.gather(self.embed_ent, self.entity_ph), [1])
        # ent = tf.expand_dims(tf.matmul(ent, Went), axis=1)
        ent = tf.expand_dims(ent, axis=1)
        ents = tf.nn.tanh(repeat(ent, times=constants.max_sent_len, axis=1))
        # ents = repeat(ent, times=constants.max_sent_len, axis=1)

        attr = tf.squeeze(tf.gather(self.embed_attr, self.attr_ph), [1])
        # attr = tf.expand_dims(tf.matmul(attr, Wattr), axis=1)
        attr = tf.expand_dims(attr, axis=1)
        attrs = tf.nn.tanh(repeat(attr, times=constants.max_sent_len, axis=1))
        # attrs = repeat(attr, times=constants.max_sent_len, axis=1)

        # (batch, sent, n_filter + ent_dim)
        concats_ent = tf.nn.tanh(tf.concat(2, [words, ents]))
        concats_attr = tf.nn.tanh(tf.concat(2, [words, attrs]))

        score_func = attn_score(self.attn_score_func)

        def attention_layer(batch_idx_and_sent_len):
            idx, sent_len = batch_idx_and_sent_len[0], batch_idx_and_sent_len[1]
            # (sent, x)
            trimmed_ent = concats_ent[idx, :sent_len, :]
            trimmed_attr = concats_attr[idx, :sent_len, :]

            # attention weights (shape depends on sentence length), (sent, 1)
            alignment_ent = score_func(tf.matmul(trimmed_ent, Wa_ent))
            alignment_attr = score_func(tf.matmul(trimmed_attr, Wa_attr))
            alignment = score_func(tf.transpose(alignment_ent + alignment_attr))

            # (1, x)
            ctx_vec = tf.matmul(alignment, words[idx, :sent_len, :])

            projected_ctx = tf.matmul(ctx_vec, Wp)
            from_last = tf.matmul(tf.expand_dims(words[idx, sent_len-1, :], [0]), Wx)

            return tf.squeeze(tf.nn.tanh(projected_ctx + from_last), [0])

        sequence = tf.stack([tf.range(self.b_size_ph), self.sent_len_ph], axis=1)
        mapped = tf.map_fn(attention_layer, sequence, dtype=tf.float32)
        logits = tf.squeeze((tf.matmul(mapped, Ws,)))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.y_ph))

        # weight decay
        # l2_norms = reduce(tf.add, [tf.nn.l2_loss(var) for var in tf.global_variables()
        #                            if not var.name.startswith('embed') and not var.name.startswith('conv')])

        optimizer = tf.train.AdamOptimizer(self.lr)
        # gradients = optimizer.compute_gradients(self.loss + self.w_decay_factor * l2_norms)
        gradients = optimizer.compute_gradients(self.loss)
        # gradients = [(tf.clip_by_value(g, -0.3, 0.3), var) for g, var in gradients]
        self.train_op = optimizer.apply_gradients(gradients)

        # metrics
        booleans = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y_ph, 1))
        self.accuracy_op = tf.reduce_mean(tf.cast(booleans, "float"))
