import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
from tensorflow.keras.optimizers import Adam, SGD
import pickle


class BaselineNN:
    def __init__(self, input_size, optimizer, load_weights=False):

        self.input_size = input_size  # input size of feature space output from Resnet
        self.optimizer = optimizer

        self.model = self._build_network()
        if load_weights:
            self.model.load_weights(LOAD_DIR)

    def _build_network(self):
        inputs = Input(shape=self.input_size)
        hd = self._dense_batch_layer(512, inputs)
        hd = self._dense_batch_layer(256, hd)
        hd = self._dense_batch_layer(128, hd)
        hd = self._dense_batch_layer(128, hd)
        hd = self._dense_batch_layer(128, hd)
        hd = self._dense_batch_layer(64, hd)
        hd = self._dense_batch_layer(64, hd)
        hd = self._dense_batch_layer(64, hd)

        outputs = Dense(12, activation='sigmoid')(hd)

        model = Model(inputs, outputs)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

    @staticmethod
    def _dense_batch_layer(num_units, prev_layer):
        hd = Dense(num_units,
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                   kernel_initializer='he_uniform')(prev_layer)
        hd = BatchNormalization()(hd)
        hd = ReLU()(hd)
        return hd

    def train_model(self, train_x, train_y, batch_size, epochs):
        self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.1, shuffle=True)
        self.model.save_weights(SAVE_DIR)

    def evaluate(self, test_x, test_y):
        results = self.model.evaluate(test_x, test_y)


LOAD_DIR = './model_weights'
SAVE_DIR = LOAD_DIR


def main():
    tr_data_file = 'feature_embedding.pkl'
    train_x = np.load(tr_data_file, allow_pickle=True)

    tr_label_file = 'testing_img_labels.pkl'
    with open(tr_label_file, 'rb') as f:
        raw_y = pickle.load(f)

    baseline_nn = BaselineNN(input_size=1024,
                             optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                             # optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                             load_weights=False)

    baseline_nn.train_model(train_x, train_y, 64, 200)

    print('break')


if __name__ == '__main__':
    main()
