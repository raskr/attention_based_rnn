import math
from nltk.tokenize import word_tokenize as tokenize
import xml.etree.ElementTree as Parser
import os
import re
import numpy as np
from logging import getLogger, StreamHandler, INFO, WARNING

import constants

logger = getLogger(__name__)
logger.addHandler(StreamHandler())


def log(msg, warn=False, ):
    logger.log(WARNING if warn else INFO, msg)


def train_val_split(seq, test_idx, fold=None,):
    """
    :param seq: any list-like
    :param fold: k-fold (optional)
    :return:
    """
    n_data = len(seq)
    fold = fold or int(1 + math.log(n_data)/math.log(2))
    print 'train : validation = {} : 1', fold - 1

    train = seq[:test_idx] + seq[test_idx+1:]
    val = seq[test_idx]

    return train, val


def pad_sequence(s, maxlen=None, value=0):
    length = len(s)
    assert length > 0
    maxlen = maxlen or length

    sample_shape = tuple(np.asarray(s).shape[1:])

    x = (np.ones((maxlen,) + sample_shape) * value).astype('int32')
    trunc = s[:maxlen]

    # check `trunc` has expected shape
    trunc = np.asarray(trunc, dtype='int32')
    if trunc.shape[1:] != sample_shape:
        raise ValueError('Shape of sample %s of sequence is different from expected shape %s' %
                         (trunc.shape[1:], sample_shape))

    x[:len(trunc)] = trunc
    return x


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0):
    """ copy from keras.preprocessing.sequence.pad_sequence()
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def to_categorical(y, nb_classes=None):
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def filter_symbol_1(sent):
    def filter_symbol_w(word):
        dst = word.rstrip('.')
        dst = dst.rstrip('-')
        # if word != dst:
        #     raise RuntimeError(' filter_symbol_w', dst, word)
        return dst
    return filter(lambda x: x != '', map(filter_symbol_w, sent))


r = re.compile(u'\.+')
def filter_symbol(sent):
    sw = [u',', u':', u';', u"''", u'"', u'\'', u'?', u'!', u')', u'(', u'*', u'\u2013', u'-']
    dst = filter(lambda w: w not in sw, sent)
    dst = filter(lambda x: not r.match(x), dst)
    return filter_symbol_1(dst)


def to_lower(sent):
    return [w.lower() if w != 'I' else w for w in sent]


def load_semeval_sents(filename):
    sents = []
    for review in Parser.parse(filename).getroot().findall('.//sentence'):
        sent = tokenize(review.find('text').text)
        sent = to_lower(sent)
        sent = filter_symbol(sent)
        sents.append(sent)
    return sents


def load_embedding_weights():
    vector_file = constants.base_path + '/saved_models/vectors{}.txt'.format(constants.year)
    assert os.path.exists(vector_file)

    with open(vector_file, 'r') as infile:
        header = str(infile.readline())
        vocab_size, vec_dim = map(int, header.strip().split())

        words, vectors = [], []
        for line in infile.readlines():
            parts = str(line).rstrip().split(" ")
            assert len(parts) == 1 + vec_dim
            word, vec = parts[0], np.array((map(np.float32, parts[1:])))
            vectors.append(vec)
            words.append(word)

    # --- words and vectors from google news were created,
    # --- but this will make cache miss on semeval dataset.

    semeval_sents = load_semeval_sents(constants.train_filename) +\
                    load_semeval_sents(constants.test_filename)

    semeval_words = []
    for sent in semeval_sents:
        semeval_words.extend(sent)

    not_covered = set(semeval_words) - set(words)
    print '{}% of words was covered'.format(100.*(1. - float(len(not_covered))/len(set(semeval_words))))
    words.extend(list(not_covered))
    vectors.extend([np.random.uniform(-1., 1., vec_dim) for _ in range(len(not_covered))])

    # index 0 is for padding vector
    word2idx = {word: i for i, word in zip(range(1, len(words) + 1), words)}
    vectors.insert(0, np.zeros((vec_dim,)))

    return word2idx, np.asarray(vectors), [word2idx[w] for w in not_covered]
