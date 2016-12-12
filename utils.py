import math
import glob
import random
import os
import re
import numpy as np
from logging import getLogger, StreamHandler, DEBUG

base_path = os.path.dirname(os.path.abspath(__file__))
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)


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


def pad_sequences_(sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0):
    """ copied from keras.preprocessing.sequence.pad_sequence()
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


def filter_symbol(sent):
    r = re.compile(u'\.+')
    sw = [u',', u':', u';', u"''", u'"', u'\'', u'?', u'!', u')', u'(', u'*', u'\u2013', u'-']
    dst = filter(lambda w: w not in sw, sent)
    dst = filter(lambda x: not r.match(x), dst)
    return filter_symbol_1(dst)


def to_lower(sent):
    return [w.lower() if w != 'I' else w for w in sent]


def list_fnames(dirname, ext=None, name_only=False, full_path=False):

    if ext:
        filenames = glob.glob('{}/*.{}'.format(dirname, ext))
    else:
        filenames = glob.glob('{}/*'.format(dirname))

    assert len(filenames) > 0, 'file not found'

    if name_only:
        return [fname.split('/')[-1] for fname in filenames]
    elif full_path:
        return [base_path + fname[1:] for fname in filenames]
    else:
        return filenames


