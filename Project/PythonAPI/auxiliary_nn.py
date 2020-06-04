import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
from tensorflow.keras.optimizers import Adam, SGD


LOAD_DIR = './model_weights/'


class AuxiliaryNN:
    def __init__(self, input_size, lr, optimizer, load_weights=False):

        self.input_size = input_size  # input size of feature space output from Resnet
        self.learning_rate = lr
        self.optimizer = optimizer

        self.model = self._build_network()
        if load_weights:
            self.model.load_weights(LOAD_DIR)

    def _build_network(self):
        inputs = Input(shape=self.input_size)
        hd = self._dense_batch_layer(128, inputs)
        hd = self._dense_batch_layer(128, hd)
        hd = self._dense_batch_layer(128, hd)
        hd = self._dense_batch_layer(128, hd)

        outputs = Dense(5, activation='softmax')(hd)

        model = Model(inputs, outputs)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
        return model

    @staticmethod
    def _dense_batch_layer(num_units, prev_layer):
        hd = Dense(num_units, kernel_initializer='he_uniform')(prev_layer)
        hd = BatchNormalization()(hd)
        hd = ReLU()(hd)
        return hd

    def train_model(self, batch_size, epochs, training_iters):
        for i in range(training_iters):
            pass

    def predict(self):
        pass



def main():
    pass


if __name__ == '__main__':
    main()
