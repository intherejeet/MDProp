from batchminer import random_distance
from batchminer import random, distance

BATCHMINING_METHODS = {'random':random,
                       'distance':distance,
                       'random_distance': random_distance,
                        }

def select(batchminername, opt):
    if batchminername not in BATCHMINING_METHODS: raise NotImplementedError('Batchmining {} not available!'.format(batchminername))

    batchmine_lib = BATCHMINING_METHODS[batchminername]

    return batchmine_lib.BatchMiner(opt)
