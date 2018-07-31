import pandas
import numpy as np

# Use pandas to read the train data file, which is much faster than numpy
# d = np.loadtxt(open("train.csv", "rb"), delimiter=',', skiprows=1)

TRAIN = "./data/train.csv"
TEST = "./data/test.csv"

class DataLoader(object):
    def __init__(self, train_file=TRAIN, test_file=TEST):
        self.train_file = train_file
        self.test_file = test_file

    def load_train_data(self, ratio=1):
        """
        Load given raito of train data.
        Return label data and feature data respectively.
        """
        self.train_data = pandas.io.parsers.read_csv(self.train_file).values
        n, _ = self.train_data.shape
        x_train = self.train_data[:int(n*ratio), 1:].reshape((-1, 1, 28, 28))
        y_train = self.train_data[:int(n*ratio), :1]
        return x_train, y_train

    def load_test_data(self, ratio=1):
        """
        Load given ratio of test data
        """
        self.test_data = pandas.io.parsers.read_csv(self.test_file).values
        n, _ = self.test_data.shape
        return self.test_data[:int(n*ratio), :].reshape((-1, 1, 28, 28))


def main():
    d = DataLoader()
    x, y = d.load_train_data(0.1)
    print x.shape, y.shape
    x_ = d.load_test_data(0.1)
    print x_.shape
    print y

if __name__ == '__main__':
    main()