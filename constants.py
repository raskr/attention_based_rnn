import os

max_sent_len = 40
word_vec_dim = 300
n_entity = 6
n_attr = 5
n_label = 3

year = 2015

base_path = os.path.dirname(os.path.abspath(__file__))

if year == 2015:
    train_filename = base_path + '/train_xml/15_res_train.xml'
    test_filename = base_path + '/test_xml/15_res_test.xml'
else:
    train_filename = base_path + '/train_xml/16_res_train.xml'
    test_filename = base_path + '/test_xml/16_res_test.xml'

# from NLTK english stopwords
stopwords = {u'all', u'just', u'being', u'over', u'both', u'through', u'yourselves',
             u'its', u'before', u'herself', u'had', u'should', u'to', u'only', u'under',
             u'ours', u'has', u'do', u'them', u'his', u'very', u'they', u'not', u'during',
             u'now', u'him', u'nor', u'did', u'this', u'she', u'each', u'further', u'where',
             u'few', u'because', u'doing', u'some', u'are', u'our', u'ourselves', u'out',
             u'what', u'for', u'while', u'does', u'above', u'between', u't', u'be', u'we',
             u'who', u'were', u'here', u'hers', u'by', u'on', u'about', u'of', u'against',
             u's', u'or', u'own', u'into', u'yourself', u'down', u'your', u'from', u'her',
             u'their', u'there', u'been', u'whom', u'too', u'themselves', u'was', u'until',
             u'more', u'himself', u'that', u'but', u'don', u'with', u'than', u'those', u'he',
             u'me', u'myself', u'these', u'up', u'will', u'below', u'can', u'theirs', u'my',
             u'and', u'then', u'is', u'am', u'it', u'an', u'as', u'itself', u'at', u'have',
             u'in', u'any', u'if', u'again', u'no', u'when', u'same', u'how', u'other', u'which',
             u'you', u'I', u'after', u'most', u'such', u'why', u'a', u'off', u'i', u'yours',
             u'so',u'the', u'having', u'once'}
