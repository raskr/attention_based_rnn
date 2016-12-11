import math
import numpy as np

def random_search():
    # from scipy.stats.distributions import expon
    # param_grid = {'a':[1, 2], 'b': expon()}

    from sklearn.model_selection import ParameterSampler
    param_dict = {
        'lr': np.exp(np.random.uniform(math.log(0.0006), math.log(0.005), 1000)),
        'b_size': [16, 32],
        'filter_len': [2, 5],
        'n_filter': [32, 64, 128, 256],
        'pool': [2, 3, 4],
        'rnn_dim': [64, 128, 256],
        'ent_dim': [32, 64, 128],
        'attr_dim': [32, 64, 128],
        'epoch': list(range(2, 18))
    }

    import numpy.random as rd
    from scipy.stats import expon
    from matplotlib import pyplot as plt
    weight_decay = 10 ** np.random.uniform(-8, -4)
    plt.hist(weight_decay)
    # plt.plot(weight_decay)

    # x = np.ones((100,)) - np.random.exponential(size=(100,))
    # plt.plot(1, 1)

    # plt.plot(x)

    # expon()
    # b = np.histogram(a)
    # plt.hist(expon.pdf(rd.uniform()))
    plt.savefig('a.png')
    # print (rd.randn(1, 10) + 1) * 0.001
    # param_list = list(ParameterSampler(param_dict, n_iter=4))
    # print param_list

random_search()
