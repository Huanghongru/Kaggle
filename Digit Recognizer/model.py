import keras
import numpy as np
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from data_loader import *

class Model(object):
    """
    Implement LeNet-5
    """
    def __init__(self, input_shape=(1, 28, 28), lr=1e3, batch_size=64, epochs=4, isTrain=True):
        self.lr = lr
        self.isTrain = isTrain
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_shape = input_shape

    def create_model(self):
        """
        """
        self.model = Sequential()
        self.model.add(Conv2D(6, kernel_size=(5, 5),
                        padding='same', activation='relu',
                        data_format="channels_first",
                        input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, kernel_size=(5, 5),
                        data_format="channels_first",
                        activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(84, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adam(lr=self.lr),
                           metrics=['accuracy'])

    def train(self, x_train, y_train, save_model=True):
        """
        """
        self.create_model()
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs)
        if save_model:
            self.model.save('./models/mnist.h5')

    def predict(self, x_test, save_csv=True):
        """
        """
        if not self.isTrain:
            self.model = keras.models.load_model('./models/mnist.h5')
        result = self.model.predict(x_test, batch_size=self.batch_size).argmax(axis=-1)

        if save_csv:
            with open("result.csv", "w") as f:
                f.write("ImageId,Label\n");
                for i, label in enumerate(result):
                    f.write("%s,%s\n" % (i+1, label))
        return result


def main():
    d = DataLoader()
    x_train, y_train = d.load_train_data(ratio=0.005)
    y_train = keras.utils.to_categorical(y_train, 10)
    lenetModel = Model()
    lenetModel.train(x_train, y_train, save_model=False)

    # x_test = d.load_test_data()
    # lenetModel.predict(x_test)

if __name__ == '__main__':
    main()
