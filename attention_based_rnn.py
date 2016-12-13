import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import constants
import numpy as np
from sklearn.base import BaseEstimator


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
                self.padding_ph: np.zeros((self.batch_size, self.filter_len//2, self.vec_dim), dtype='float32')
               }

    def eval(self, sess, ids, entity, attr, sent_lens, labels):
        feed_dict = \
            {self.ids_ph: ids,
             self.entity_ph: np.expand_dims(entity, 1),
             self.attr_ph: np.expand_dims(attr, 1),
             self.sent_len_ph: sent_lens/self.pool_len,
             self.pool_len_ph: self.pool_len,
             self.b_size_ph: len(ids),
             self.padding_ph: np.zeros((len(ids), self.filter_len//2, self.vec_dim), dtype='float32'),
             self.y_ph: labels}

        return self.accuracy_op.eval(feed_dict, sess)

    def build_graph(self, n_ent, n_attr, vec_dim, label_num):

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

        def conv1d(x, W, b, pool_len=1, padding=None):
            # x = (batch, sent, vec_dim)
            if padding is not None:
                x = tf.concat(1, (padding, x, padding))
            x = tf.expand_dims(x, 3)
            x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
            # max_pool is effective!!
            if pool_len != 1:
                x = tf.nn.max_pool(x, ksize=[1, pool_len, 1, 1], strides=[1, pool_len, 1, 1], padding='VALID')
            x = tf.squeeze(x, [2])
            x = tf.nn.bias_add(x, b)
            return x

        def attn_score(string):
            if string == 'sigmoid':
                return tf.nn.sigmoid
            elif string == 'h_sigmoid':
                return hard_sigmoid
            elif string == 'softmax':
                return tf.nn.softmax

        def xavier_init(shape, name):
            return tf.get_variable(name, shape=shape, initializer=xavier_initializer(), dtype=tf.float32)

        # Define operations and tensors in `graph`.
        # ------------
        # placeholders
        # ------------
        self.b_size_ph = tf.placeholder(shape=(), dtype=tf.int32, name='b_size')
        self.pool_len_ph = tf.placeholder(shape=(), dtype=tf.int32, name='pool_len')
        self.embedding_w_ph = tf.placeholder(shape=(self.n_vocab, vec_dim), dtype=tf.float32, name='w')
        self.embedding_e_ph = tf.placeholder(shape=(n_ent, vec_dim), dtype=tf.float32, name='e')
        self.embedding_a_ph = tf.placeholder(shape=(n_attr, vec_dim), dtype=tf.float32, name='a')
        self.ids_ph = tf.placeholder(shape=(None, constants.max_sent_len,), dtype=tf.int32, name='ids')
        self.entity_ph = tf.placeholder(shape=(None, 1,), dtype=tf.int32, name='ent')
        self.attr_ph = tf.placeholder(shape=(None, 1,), dtype=tf.int32, name='attr')
        self.y_ph = tf.placeholder(shape=(None, label_num), dtype=tf.int32, name='y')
        self.sent_len_ph = tf.placeholder(tf.int32, shape=(None,), name='sent_len')
        self.padding_ph = tf.placeholder(shape=(None, self.filter_len//2, vec_dim), dtype=tf.float32, name='pad')

        # -----
        # param
        # -----
        if self.use_convolution:
            sent_len_after_conv = constants.max_sent_len/self.pool_len
        else:
            sent_len_after_conv = constants.max_sent_len

        # ---------
        # variables
        # ---------
        # for embedding (values will be override by `assign`)
        self.embed_tok = tf.Variable(tf.random_uniform((self.n_vocab, vec_dim)))
        self.embed_ent = tf.Variable(tf.random_uniform((n_ent, vec_dim)))
        self.embed_attr = tf.Variable(tf.random_uniform((n_attr, vec_dim)))
        # for aspect projection
        Went = xavier_init((vec_dim, self.ent_vec_dim), name='Went')
        Wattr = xavier_init((vec_dim, self.attr_vec_dim), name='Wattr')
        # bv = tf.Variable(tf.zeros((self.rnn_dim,)))
        # for attention
        Wa_ent = xavier_init((self.rnn_dim + self.ent_vec_dim, 1), name='Wa_ent')
        Wa_attr = xavier_init((self.rnn_dim + self.attr_vec_dim, 1), name='Wa_attr')
        # for conv
        Wc = xavier_init((self.filter_len, vec_dim, 1, self.n_filter), name='Wc')
        bc = tf.Variable(tf.zeros((self.n_filter,)))
        # before prediction
        Wx = xavier_init((self.rnn_dim, self.rnn_dim), name='Wx')
        Wp = xavier_init((self.rnn_dim, self.rnn_dim), name='Wp')
        # for prediction
        Ws = xavier_init((self.rnn_dim, label_num), name='Ws')
        bs = tf.Variable(tf.zeros((label_num,)))
        if self.rnn_unit == 'BasicLSTM':
            rnn_unit = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_dim)
        elif self.rnn_unit == 'LSTM':
            rnn_unit = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_dim, cell_clip=self.cell_clip)
        elif self.rnn_unit == 'GRU':
            rnn_unit = tf.nn.rnn_cell.GRU(num_units=self.rnn_dim)

        # ----
        # flow
        # ----
        self.embedding_w_init_op = tf.assign(self.embed_tok, self.embedding_w_ph)
        self.embedding_e_init_op = tf.assign(self.embed_ent, self.embedding_e_ph)
        self.embedding_a_init_op = tf.assign(self.embed_attr, self.embedding_a_ph)

        # (batch, 70, vec_dim)
        words = tf.gather(self.embed_tok, self.ids_ph)

        if self.use_convolution:
            words = conv1d(words, Wc, bc, pool_len=self.pool_len, padding=self.padding_ph)

        # (batch, sent, 128)
        hs, _ = tf.nn.dynamic_rnn(rnn_unit, words, sequence_length=self.sent_len_ph, dtype=tf.float32)

        if not self.use_attention:
            logits = tf.matmul(hs[:, -1, :], Ws,) + bs
        else:
            # (batch, cat_vec_dim)
            # ent = self.embed_ent[self.entity_ph, :]
            ent = tf.squeeze(tf.gather(self.embed_ent, self.entity_ph), [1])
            attr = tf.squeeze(tf.gather(self.embed_attr, self.attr_ph), [1])

            # (batch, rnn_dim)
            ent = tf.expand_dims(tf.matmul(ent, Went), axis=1) # + self.bv
            attr = tf.expand_dims(tf.matmul(attr, Wattr), axis=1) # + self.bv
            ents = tf.tanh(repeat(ent, times=sent_len_after_conv, axis=1))
            attrs = tf.tanh(repeat(attr, times=sent_len_after_conv, axis=1))

            # (batch, sent, vec_dim + ent_dim)
            concat_ent = tf.concat(2, [ents, hs])
            concat_attr = tf.concat(2, [attrs, hs])

            score_func = attn_score(self.attn_score_func)

            # core flow
            def compute_ctx_vec(batch_idx_and_sent_len):
                idx, sent_len = batch_idx_and_sent_len[0], batch_idx_and_sent_len[1]
                trimmed_ent = concat_ent[idx, :sent_len, :]
                trimmed_attr = concat_attr[idx, :sent_len, :]
                attn_ent = score_func(tf.matmul(trimmed_ent, Wa_ent))
                attn_attr = score_func(tf.matmul(trimmed_attr, Wa_attr))
                # (1, sent)
                attn = tf.transpose(attn_ent * attn_attr)
                # (sent, rnn_dim)
                sliced_hs = hs[idx, :sent_len, :]
                # (1, rnn_dim)
                ctx_vec = tf.matmul(attn, sliced_hs)

                a = tf.matmul(ctx_vec, Wp)
                b = tf.matmul(tf.slice(sliced_hs, [sent_len-1, 0], [1, -1]), Wx)
                dst = tf.nn.tanh(a + b)
                return tf.squeeze(dst, [0])

            sequence = tf.stack([tf.range(self.b_size_ph), self.sent_len_ph], axis=1)
            context_vec = tf.map_fn(compute_ctx_vec, sequence, dtype=tf.float32) # (batch, rnn_dim)
            logits = tf.squeeze((tf.matmul(context_vec, Ws,) + bs)) # (batch, 3)

        sce = tf.nn.softmax_cross_entropy_with_logits(logits, self.y_ph)

        # weight decay
        l2_norms = \
            tf.nn.l2_loss(Went) + \
            tf.nn.l2_loss(Wattr) + \
            tf.nn.l2_loss(Wa_ent) + \
            tf.nn.l2_loss(Wa_attr) + \
            tf.nn.l2_loss(Wx) + \
            tf.nn.l2_loss(Wp) + \
            tf.nn.l2_loss(Ws)

        self.loss = tf.reduce_mean(sce) + self.w_decay_factor * l2_norms
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        # metrics
        booleans = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y_ph, 1))
        self.accuracy_op = tf.reduce_mean(tf.cast(booleans, "float"))
