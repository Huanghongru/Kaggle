import keras
import numpy as np
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from data_loader import *

class Model(object):
    """
    Implement LeNet-5
    """
    def __init__(self, input_shape=(28, 28, 1), lr=1e-3, 
                batch_size=64, epochs=4, isTrain=True, aug_train=False):
        self.lr = lr
        self.isTrain = isTrain
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_shape = input_shape
        self.aug_train = aug_train

        self.datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False)


    def create_model(self):
        """
        """
        self.model = Sequential()
        self.model.add(Conv2D(6, kernel_size=(5, 5),
                        padding='same', activation='relu',
                        input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, kernel_size=(5, 5),
                        activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(84, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adam(lr=self.lr),
                           metrics=['accuracy'])

    def train(self, x_train, y_train, x_val, y_val, save_model=True, verbose=1):
        """
        """
        self.create_model()
        if self.aug_train:
            self.model.fit_generator(self.datagen.flow(x_train, y_train, 
                                   batch_size=self.batch_size),
                                   verbose=verbose,
                                   validation_data = (x_val, y_val),
                                   epochs=self.epochs)
            if save_model:
                self.model.save('./models/mnist.h5')
        else: 
            self.model.fit(x_train, y_train,
                           batch_size=self.batch_size,
                           validation_split=0.2,
                           verbose=verbose,
                           epochs=self.epochs)
            if save_model:
                self.model.save('./models/mnist.h5')

    def evaluate(self, x_val, y_val, verbose=1):
        """
        """
        if not self.isTrain:
            self.model = keras.models.load_model('./models/mnist.h5')
        r = self.model.evaluate(x_val, y_val, verbose=verbose)
        val_loss, val_acc = r[0], r[1]
        return val_loss, val_acc

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
    x_train, y_train = d.load_train_data(ratio=0.2)
    y_train = keras.utils.to_categorical(y_train, 10)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    lenetModel = Model(aug_train=True)
    lenetModel.train(x_train, y_train, x_val, y_val, save_model=False)

    # x_test = d.load_test_data()
    # lenetModel.predict(x_test)

if __name__ == '__main__':
    main()
