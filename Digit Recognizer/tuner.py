import keras
import random
import numpy as np 
from model import *
from data_loader import *

def tune_lr_batchSize(lr_range, bs_range, data_ratio=0.005, tune_epoch=100):
    """
    Randomly sample learning rate and batch_size within a range.
    """
    d = DataLoader()
    x, y = d.load_train_data(ratio=data_ratio)

    n = x.shape[0]
    x_train = x[:int(n*0.8),:,:,:]
    y_train = y[:int(n*0.8)]
    y_train = keras.utils.to_categorical(y_train, 10)
    x_val = x[int(n*0.8):,:,:,:]
    y_val = y[int(n*0.8):]
    y_val = keras.utils.to_categorical(y_val, 10)

    for i in range(tune_epoch):
        lr = random.uniform(lr_range[0], lr_range[1])
        batch_size = random.choice(bs_range)

        model = Model(lr=lr, batch_size=batch_size)
        model.train(x_train, y_train, save_model=False, verbose=0)
        loss, acc = model.evaluate(x_val, y_val, verbose=0)

        print "learning rate: %.5f\tbatch_size: %s\t" % (lr, batch_size),
        print "loss: %.3f\tacc: %.3f\t" % (loss, acc),
        print "\t( %s / %s )\n" % (i+1, tune_epoch)

def main():
    tune_lr_batchSize((1e-5, 1e-2), [32, 64, 128, 256], data_ratio=0.02)

if __name__ == '__main__':
    main()
