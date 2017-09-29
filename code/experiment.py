from __future__ import division, print_function
import math
from os.path import join, isdir
from experimenter.utilities import ExperimentDataset, Experiment

caffemodel=None
datadir = '/home/bchu/CS280MiniPlaces/code/data/'
basedir = '/home/bchu/CS280MiniPlaces/runs/'

train_batch_size = 256
test_batch_size = 250
val_batch_size = 250
num_classes = 100

dataset= ExperimentDataset(name='all', min_epochs=200, min_iters=1200, cycle_iters=300,
                           train_path=datadir + 'train.txt', val_path=datadir + 'val.txt')


def generate(group):    
    return [ 
        Experiment(folder=join(basedir, 'mini-alexnet'), train_iters=80000,
                   name='mini-alexnet', dataset=dataset),
        Experiment(folder=join(basedir, 'alexnet'), train_iters=80000,
                   name='alexnet', dataset=dataset),
        Experiment(folder=join(basedir, 'alexnet-batchnorm'), train_iters=80000,
                   name='alexnet-batchnorm', dataset=dataset),
        Experiment(folder=join(basedir, 'scene-stack'), train_iters=80000,
                   name='scene-stack', dataset=dataset),
        Experiment(folder=join(basedir, 'alexnet-stack-2'), train_iters=80000,
                   name='alexnet-stack-2', dataset=dataset),
        Experiment(folder=join(basedir, 'alexnet-linear'), train_iters=80000,
                   name='alexnet-linear', dataset=dataset),
    ]