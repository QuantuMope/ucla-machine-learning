import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.preprocessing import LabelEncoder


LOAD_DIR = './model_weights/'


class BaselineNN:
    def __init__(self, input_size, lr, optimizer, load_weights=False):

        self.input_size = input_size  # input size of feature space output from Resnet
        self.learning_rate = lr
        self.optimizer = optimizer

        self.model = self._build_network()
        if load_weights:
            self.model.load_weights(LOAD_DIR)

        labels = ['gender', 'truck', 'motorcycle', 'tie', 'backpack', 'sports ball',
                  'handbag', 'fork', 'knife', 'spoon', 'cell phone', 'teddy bear']

        self.le = LabelEncoder()
        self.le.fit(labels)

    def _build_network(self):
        inputs = Input(shape=self.input_size)
        hd = self._dense_batch_layer(128, inputs)
        hd = self._dense_batch_layer(128, hd)
        hd = self._dense_batch_layer(128, hd)
        hd = self._dense_batch_layer(128, hd)

        outputs = Dense(12, activation='softmax')(hd)

        model = Model(inputs, outputs)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

    @staticmethod
    def _dense_batch_layer(num_units, prev_layer):
        hd = Dense(num_units, kernel_initializer='he_uniform')(prev_layer)
        hd = BatchNormalization()(hd)
        hd = ReLU()(hd)
        return hd

    def train_model(self, train_x, train_y, batch_size, epochs):
        self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)

    def evaluate(self, test_x, test_y):
        results = self.model.evaluate(test_x, test_y)


def main():
    pass


if __name__ == '__main__':
    main()
