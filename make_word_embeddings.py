import os.path
import sys
from nltk.tokenize import word_tokenize as tokenize
import xml.etree.ElementTree as Parser

import utils
import constants


def load_semeval_sents(filename):
    sents = []
    for review in Parser.parse(filename).getroot().findall('.//sentence'):
        sent = tokenize(review.find('text').text)
        sent = utils.to_lower(sent)
        sent = utils.filter_symbol(sent)
        sents.append(sent)
    return sents


def make_vectors_from_google_news(trained_vector_file_path,
                                  train_filename,
                                  test_filename,
                                  output_filename):
    import gzip
    import numpy as np

    print 'read vectors'
    with gzip.open(trained_vector_file_path, 'rb') as infile:
        mapping = {}
        header = str(infile.readline())
        vocab_size, vec_dim = map(int, header.split())

        binary_len = np.dtype(np.float32).itemsize * vec_dim
        for line_no in xrange(vocab_size):
            # mixed text and binary: read text first, then binary
            word = []
            while True:
                ch = infile.read(1)
                if ch == b' ':
                    break
                if ch == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                    word.append(ch)

            word = str(b''.join(word))
            weights = np.fromstring(infile.read(binary_len), dtype=np.float32)
            mapping[word] = weights

    print 'write vectors to file'
    # get vectors from `google vectors`
    sents1 = load_semeval_sents(train_filename)
    sents2 = load_semeval_sents(test_filename)

    words = []
    for sent in sents1 + sents2:
            words.extend(sent)
    words = set(words)

    with open(output_filename, 'w') as outfile:
        # write header
        outfile.write("{} {}\n".format(len(words), vec_dim).encode('utf-8'))

        # write actual data
        for word in words:
            if word in mapping:
                vec = mapping[word]
            else:
                vec = np.random.uniform(-1., 1., vec_dim)
            outfile.write(("%s %s\n" % (word, ' '.join("%f" % val for val in vec))).encode('utf8'))
        print 'vectors{}.txt has been created in saved_models/'.format(constants.year)


if not os.path.exists(constants.base_path + '/GoogleNews-vectors-negative300.bin.gz'):
    print '### Downloading ./GoogleNews-vectors-negative300.bin.gz\n'
    sys.stdout.flush()
    os.system('curl -O https://s3.amazonaws.com/mordecai-geo/GoogleNews-vectors-negative300.bin.gz')

make_vectors_from_google_news(
        constants.base_path + '/GoogleNews-vectors-negative300.bin.gz',
        constants.train_filename,
        constants.test_filename,
        constants.base_path + '/saved_models/vectors{}.txt'.format(constants.year))
