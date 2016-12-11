import utils
import numpy as np
import json
from operator import add
from nltk.tokenize import word_tokenize as tokenize
from gensim import models


def load_yelp_sents(json_filename, yelp_range):
    """
    :param json_filename:
    :return: string list
    """
    assert isinstance(yelp_range, list)

    def process(line):
        review_text = json.loads(line)['text']
        sents = review_text.split()
        sents = [utils.filter_symbol(utils.to_lower(tokenize(sent))) for sent in sents]
        return sents

    with open(json_filename, 'r') as in_file:
        dst = []
        lines = in_file.readlines()[yelp_range[0]: yelp_range[1]]
        all_len = len(lines)
        for i, line in enumerate(lines):
            sents = process(line)
            dst.extend(sents)
            if i % 40000 == 0:
                print 'processed {}'.format(float(i)/all_len)
        return dst


def load_semeval_sents(filename):
    """
    :param filename: path to the xml file contains ABSA Review data
    :return: list of (list of string)
    """

    import xml.etree.ElementTree as Parser

    sents = []

    for review in Parser.parse(filename).getroot().findall('.//sentence'):
        # string list
        sent = tokenize(review.find('text').text)
        sent = utils.to_lower(sent)
        sent = utils.filter_symbol(sent)
        sents.append(sent)

    return sents


def train_with_test_data(w2v_model, tf_weight):
    # -1 mean row for padding
    w2v_model.syn0 = tf_weight[:w2v_model.syn0.shape[0], :]
    sents = load_semeval_sents(utils.base_path + '/test_xml/EN_REST_SB1_TEST.xml')
    w2v_model.build_vocab(sentences=sents, update=True)
    w2v_model.train(sents)
    return w2v_model


def train_chunk(trained_model, yelp_range,):
    # build new gensim model
    if trained_model is None:
        # load 2016 sents
        latest = load_semeval_sents(utils.list_fnames('train_xml')[0])

        # load extra sents
        past = reduce(add, [load_semeval_sents(f) for f in utils.list_fnames(dirname='semeval_past_xmls')])

        # filtering for `extra` sents
        model = models.word2vec.Word2Vec(sentences=latest + past, min_count=0, iter=8,)
        model.save_word2vec_format()
        return model

    # update existing model with extra data
    else:
        sents = load_yelp_sents(utils.list_fnames(dirname='yelp_json_file')[0], yelp_range)
        model = models.word2vec.Word2Vec(min_count=3)
        model.build_vocab(sents)
        sents = [utils.filter_symbol([w for w in sent if w in model.vocab]) for sent in sents]
        trained_model.build_vocab(sentences=sents, update=True)
        trained_model.train(sents)
        return trained_model


if __name__ == '__main__':

    trained_model = None
    max_sent_of_extra = 1000000
    step = 330000

    # train with production data
    trained_model = train_chunk(trained_model, yelp_range=None)

    # train_chunk() loop
    for i in range(0, max_sent_of_extra, step):
        print 'Trained with {}/{} reviews'.format(i + step, max_sent_of_extra)
        print 'Current model has {} vocab\n'.format(len(trained_model.syn0))
        # save regularly
        with open(utils.base_path + '/saved_models/w2v_gensim_model_new.w2v', 'w') as f:
            trained_model.save(f)
