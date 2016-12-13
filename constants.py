import os

max_sent_len = 40
word_vec_dim = 300
n_entity = 6
n_attr = 5
n_label = 3

year = 2016

base_path = os.path.dirname(os.path.abspath(__file__))

if year == 2015:
    train_filename = base_path + '/train_xml/15_res_train.xml'
    test_filename = base_path + '/test_xml/15_res_test.xml'
else:
    train_filename = base_path + '/train_xml/16_res_train.xml'
    test_filename = base_path + '/test_xml/16_res_test.xml'
