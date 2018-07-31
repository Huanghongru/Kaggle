import keras
import random
import numpy as np 
from model import *
from data_loader import *

def tune_lr_batchSize(lr_range, bs_range, tune_epoch=100):
    """
    Randomly sample learning rate and batch_size within a range.
    """
    for i in range(tune_epoch):
        lr = random.uniform(lr_range[0], lr_range[1])
        bs_range = random.choice(bs_range)

        printf("learning rate:%s batch_size: %s\n" % (lr, bs_range))

def main():
    tune_lr_batchSize((1e-5, 1e-2), [32, 64, 128, 256])

if __name__ == '__main__':
    main()
