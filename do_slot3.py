import tensorflow as tf
from sklearn.model_selection import ParameterSampler
import math
import word_embedding_gensim
import numpy as np
from review import load_semeval_reviews
from review import make_ent_attr_lookup
import utils
import constants
import attention_based_rnn
from gensim.models import word2vec
from sklearn.model_selection import RandomizedSearchCV
train_filename = 'train_xml/ABSA16_Restaurants_Train_SB1_v2.xml'
test_filename = 'test_xml/EN_REST_SB1_TEST.xml'
gensim_model = word2vec.Word2Vec.load(utils.base_path + '/saved_models/w2v_first114808.gensim')
# mapping word -> idx
max_vocab = int(len(gensim_model.vocab) * 1.1)
padding_idx = len(gensim_model.vocab)


def make_vocab(reviews, reusable_vocab=None):
    """
    vocab[0] raises Exception.
    idx=0 is padding value
    """
    vocab = reusable_vocab or {}
    idx = 1 if reusable_vocab is None else len(reusable_vocab)
    for review in reviews:
        for tok in review.tokens:
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def train_fool(sess):
    reviews, ent2idx, attr2idx, polarity2idx = load_semeval_reviews(train_filename, is_test_file=False)

    rows = []
    for review in reviews:
        ids = [gensim_model.vocab[tok].index for tok in review.tokens]
        sent_len = len(ids) if len(ids) < constants.max_sent_len else constants.max_sent_len
        ids = list(utils.pad_sequence(ids, constants.max_sent_len, padding_idx))
        rows.extend([np.concatenate([ids + [ent2idx[op.ent],
                                            attr2idx[op.attr],
                                            sent_len,
                                            polarity2idx[op.polarity]]], axis=-1) for op in review.opinions])

    inputs = np.array(rows, dtype='int32')
    x = inputs[:, :-1]
    y = inputs[:, -1]

    additional_rows = np.random.uniform(-1., 1., size=(max_vocab-gensim_model.syn0.shape[0], constants.word_vec_dim))
    attRNN = attention_based_rnn.AttentionBasedRNN(sess=sess,
                                                   vec_dim=constants.word_vec_dim,
                                                   n_vocab=max_vocab,
                                                   n_ent=len(ent2idx),
                                                   n_attr=len(attr2idx),
                                                   max_sent_len=constants.max_sent_len,
                                                   embedding_weight=np.vstack((gensim_model.syn0, additional_rows)))

    param_dict = {
        'lr': np.exp(np.random.uniform(math.log(0.0006), math.log(0.005), 100)),
        'epoch': list(range(32)),
        'activation': ['hard_sigmoid', 'softmax', 'sigmoid'],
        'rnn_dim': [32, 64, 128, 256],
        'batch_size': [8, 16, 32, 64, 128],
        'n_filter': [16, 32, 64, 128, 258],
        'ent_vec_dim': [8, 16, 32, 64, 128],
        'attr_vec_dim': [8, 16, 32, 64, 128],
        'pool_len': [3],
        'filter_len': [5, 7]}

    RandomizedSearchCV(attRNN, param_dict, n_iter=2, scoring=attRNN.eval).fit(x, y)

    return attRNN


def update_embedding(attLSTM, sess):
    tf_weight = np.array(attLSTM.embed_tok.eval(sess))
    new_gensim_model = word_embedding_gensim.train_with_test_data(gensim_model, tf_weight)
    # for w in gensim_model.vocab.keys():
    #     assert new_gensim_model.vocab[w].index == gensim_model.vocab[w].index
    # add trained padding-row
    tf_weight = np.vstack((new_gensim_model.syn0, tf_weight[padding_idx, :]))
    additional_rows_ = np.random.uniform(-1., 1., size=(max_vocab - tf_weight.shape[0], constants.word_vec_dim))
    tf_weight = np.squeeze(np.vstack((tf_weight, additional_rows_)))
    sess.run(attLSTM.embedding_init_op, feed_dict={attLSTM.embedding_w_ph: tf_weight})
    global gensim_model
    gensim_model = new_gensim_model


def test(attRNN):

    reviews, ent2idx, attr2idx, polarity2idx = load_semeval_reviews(test_filename, is_test_file=True)

    # vocab = make_vocab(reviews, reusable_vocab=train_vocab)
    # print len(vocab), 'vocab'

    # list of (ids, ent, attr, pol)
    tuples = []
    for review in reviews:
        ids = [gensim_model.vocab[tok].index for tok in review.tokens]
        tuples_ = [(ids, ent2idx[op.ent], attr2idx[op.attr], polarity2idx[op.polarity]) for op in review.opinions]
        tuples.extend(tuples_)

    unzipped = zip(*tuples)
    ids = utils.pad_sequences(unzipped[0], maxlen=constants.max_sent_len, value=padding_idx)
    sent_lens = np.array(map(lambda x: len(x) if len(x) < constants.max_sent_len else constants.max_sent_len, unzipped[0]), dtype='int32')
    # sent_lens /= 2
    ents, attrs, pols = (np.array(x, dtype='int32') for x in unzipped[1:])

    acc = attRNN.eval(ids, ents, attrs, sent_lens, pols)
    print 'test accuracy', acc


def random_search_cv(n_iter):
    reviews, ent2idx, attr2idx, polarity2idx = load_semeval_reviews(train_filename, is_test_file=False)

    ent_lookup, attr_lookup = make_ent_attr_lookup(reviews, gensim_model.vocab,
                                                   lambda x: gensim_model.syn0[x], ent2idx, attr2idx)

    model = attention_based_rnn.AttentionBasedRNN(n_vocab=max_vocab)

    # list of (ids, ent, attr, pol)
    tuples = []
    for review in reviews:
        ids = [gensim_model.vocab[tok].index for tok in review.tokens]
        tuples_ = [(ids, ent2idx[op.ent], attr2idx[op.attr], polarity2idx[op.polarity]) for op in review.opinions]
        tuples.extend(tuples_)

    unzipped = zip(*tuples)
    ids = utils.pad_sequences_(unzipped[0], maxlen=constants.max_sent_len)
    ents, attrs, pols = [np.array(x, dtype='int32') for x in unzipped[1:]]
    sent_lens = np.array(map(lambda x: len(x) if len(x) < constants.max_sent_len else constants.max_sent_len, unzipped[0]), dtype='int32')

    def cv(hyper_params):
        print hyper_params
        fold = 2

        perm = np.random.permutation(len(ids))
        chunk_size = len(perm)/fold
        chunks = np.split(perm, [chunk_size * (i+1) for i, a in enumerate(range(fold))])

        acc_total = 0.
        model.set_params(**hyper_params)

        graph = tf.Graph()
        with graph.as_default():
            # define variables and ops in `graph`
            model.build_graph(constants.n_entity, constants.n_attr, constants.word_vec_dim, constants.n_label)
            init_op = tf.global_variables_initializer()

        # actual cross validation
        for i in range(fold):
            test_indices = chunks[i]
            train_indices = np.array(list(set(perm) - set(test_indices)))

            ids_train, ids_val = ids[train_indices, :], ids[test_indices, :]
            ent_train, ent_val = ents[train_indices], ents[test_indices]
            attr_train, attr_val = attrs[train_indices], attrs[test_indices]
            pol_train, pol_val = pols[train_indices], pols[test_indices]
            lens_train, lens_val = sent_lens[train_indices], sent_lens[test_indices]
            additional_rows = np.random.uniform(-1., 1., size=(max_vocab-len(gensim_model.syn0), constants.word_vec_dim))

            with tf.Session(graph=graph).as_default() as sess:
                sess.run(init_op)
                # train
                model.fit(sess, ids_train, ent_train, attr_train,
                          lens_train, utils.to_categorical(pol_train),
                          np.vstack((gensim_model.syn0, additional_rows)),
                          ent_lookup, attr_lookup)

                # validate
                acc = model.eval(sess, ids_val, ent_val, attr_val, lens_val, utils.to_categorical(pol_val))
                print 'validation {}/{} acc: {}'.format(i+1, fold, acc)
                acc_total += acc

        return acc_total / fold

    sampler = ParameterSampler({
        'attn_score_func': ['hard_sigmoid', 'softmax', 'sigmoid'],
        'lr': np.exp(np.random.uniform(math.log(0.0006), math.log(0.005), 1000)),
        'w_decay_factor': [10 ** np.random.uniform(-8, -4) for _ in range(1000)],
        'rnn_dim': [32, 64, 128, 256],
        'batch_size': [8, 16, 32, 64],
        'n_filter': [32, 64, 128, 258],
        'ent_vec_dim': [8, 16, 32, 64, 128],
        'attr_vec_dim': [8, 16, 32, 64, 128],
        'pool_len': [1, 2, 3],
        'filter_len': [3, 5, 7]}, n_iter=n_iter)

    scores = np.array(map(cv, sampler))

    # print all result
    for i, param in enumerate(sampler):
        print scores[i], param

    # print best result
    import pickle
    print 'best params:', scores.argmax(), list(sampler)[scores.argmax()]
    with open('best_params.pkl', mode='wb') as f:
        pickle.dump(list(sampler)[scores.argmax()], f)

if __name__ == '__main__':
    random_search_cv(n_iter=3)
