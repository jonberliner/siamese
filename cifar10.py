import numpy as np
from bunch import Bunch
import os
import pdb

def load_cifar_10(DATA_DIR):
    # DATA_DIR = '/data/CIFAR/cifar-10-batches-py/'

    dat = [np.load(DATA_DIR + '/data_batch_%d' % (ii)) for ii in xrange(1, 6)]

    x = (np.concatenate([d['data'] for d in dat]) / 255.).astype(np.float32)
    y = np.concatenate([d['labels'] for d in dat])

    tdat = np.load(DATA_DIR + '/test_batch')
    tx = (tdat['data'] / 255.).astype(np.float32)
    ty = np.array(tdat['labels'])

    dataset = Bunch()
    dataset.train = Bunch(images=x, labels=y, num_examples=x.shape[0])
    dataset.test = Bunch(images=tx, labels=ty, num_examples=tx.shape[0])

    return dataset
