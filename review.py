import xml.etree.ElementTree as Parser
import utils
from nltk.tokenize import word_tokenize as tokenize


class Opinion:
    def __init__(self, target, category, polarity, frm, to):
        self.target = target
        self.category = category
        self.ent = category.split('#')[0]
        self.attr = category.split('#')[1]
        self.polarity = polarity
        self.frm = frm
        self.to = to


class Review:
    def __init__(self, string, opinions):
        self.string = string
        self.opinions = opinions
        self.tokens = utils.filter_symbol(utils.to_lower(tokenize(string)))
        self.ids = None

    # def get_entity_vector(self, entity):
    #     """
    #     we don't know which words are related to the entity.
    #     Given: (entity name, review text)
    #
    #     :param entity: string
    #     :return:
    #     """
    #     ent_exists = len(filter(lambda op: op.ent == entity, self.opinions)) > 0


    # def get_attr_vector(self, attr):


def load_semeval_reviews(filename, is_test_file):

    reviews = []

    sents = Parser.parse(filename).getroot().findall('.//sentence')
    print 'len of <sentences>:', len(sents)

    for i, sent in enumerate(sents):
        text = sent.find('text').text
        opinions = sent.find('Opinions')

        if opinions is None:
            continue

        ops = []
        for op in opinions.findall('Opinion'):
            ops.append(Opinion(op.get('target'),
                               op.get('category'),
                               op.get('polarity'),
                               int(op.get('from')),
                               int(op.get('to'))))

        if len(ops) >= 1:
            reviews.append(Review(text, ops))

    ents = set()
    attrs = set()
    pols = set()
    for review in reviews:
        for opinion in review.opinions:
            ents.add(opinion.ent)
            attrs.add(opinion.attr)
            pols.add(opinion.polarity)

    return reviews, \
           {v: k for k, v in enumerate(ents)}, \
           {v: k for k, v in enumerate(attrs)}, \
           {v: k for k, v in enumerate(pols)},\



def ent_attr_to_words(reviews, word2idx, not_covered):
    from nltk.corpus import stopwords
    from collections import defaultdict
    import constants

    stopwords = constants.stopwords

    # e.g. {'FOOD': set(4, 6, 8)}
    ent_map = defaultdict(set)
    # e.g. {'QUALITY': set(2, 6, 9)}
    attr_map = defaultdict(set)

    for review in reviews:
        # strings -> ids for performance
        review.ids = [word2idx[tok] for tok in review.tokens if tok not in stopwords]

        # extract entities and attributes
        ents, attrs = set(), set()
        for opinion in review.opinions:
            ents.add(opinion.ent)
            attrs.add(opinion.attr)

        # add ids to ent_map
        for ent in ents:
            for id_ in review.ids:
                if id_ not in not_covered:
                    ent_map[ent].add(id_)

        # add ids to attr_map
        for attr in attrs:
            for id_ in review.ids:
                if id_ not in not_covered:
                    attr_map[attr].add(id_)

    for k, v in ent_map.items():
        print k, 'contains', len(v), 'words'

    for k, v in attr_map.items():
        print k, 'contains', len(v), 'words'

    print '\n'

    return ent_map, attr_map


def make_ent_attr_lookup(reviews, word2idx, id2vec, ent2idx, attr2idx, not_covered):
    import numpy as np
    from operator import itemgetter
    # e.g. {'FOOD': set(4, 6, 8)}
    ent2words, attr2words = ent_attr_to_words(reviews, word2idx, not_covered)

    # sort by ent/attr ID
    # [(X1, vec), (X2, vec), ...]
    pairs1 = sorted([(ent2idx[e], reduce(np.add, map(id2vec, ws)))
                     for e, ws in ent2words.items()], key=itemgetter(0))

    pairs2 = sorted([(attr2idx[a], reduce(np.add, map(id2vec, ws)))
                     for a, ws in attr2words.items()], key=itemgetter(0))

    return np.array([pair[1] for pair in pairs1]), \
           np.array([pair[1] for pair in pairs2])
