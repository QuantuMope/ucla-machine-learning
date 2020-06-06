import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.math import log, reduce_sum
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.preprocessing import LabelEncoder

# Input size: 1024

LOAD_DIR = './model_weights/'
BATCH_SIZE = 1111  # must be global as custom loss is staticmethod
LAMBDA_ = 0.10  # auxiliary loss coefficient


class AuxiliaryNN:
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

        classification_output = Dense(12, activation='softmax')(hd)
        expected_bias_output = Dense(1, activation='linear')(hd)
        outputs = Concatenate()([classification_output, expected_bias_output])

        model = Model(inputs, outputs)
        # custom loss needed here
        model.compile(loss=self._auxiliary_loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        return model

    @staticmethod
    def _auxiliary_loss_function(true_y, pred_y):
        # May result in problems due to large difference in scaling between
        # classification and regression output
        class_true_y = true_y[:-1]
        class_pred_y = pred_y[:-1]
        regre_true_y = true_y[-1]
        regre_pred_y = pred_y[-1]

        # cross-entropy loss
        main_loss = -reduce_sum(class_true_y * log(class_pred_y))
        # mean absolute loss
        auxiliary_loss = reduce_sum(tf.abs(regre_true_y - regre_pred_y)) / BATCH_SIZE
        total_loss = main_loss + LAMBDA_ * auxiliary_loss
        return total_loss

    @staticmethod
    def _dense_batch_layer(num_units, prev_layer):
        hd = Dense(num_units, kernel_initializer='he_uniform')(prev_layer)
        hd = BatchNormalization()(hd)
        hd = ReLU()(hd)
        return hd

    def train_model(self, train_x, train_y, epochs):
        self.model.fit(train_x, train_y, epochs=epochs, batch_size=BATCH_SIZE)

    def evaluate(self, test_x, test_y):
        results = self.model.evaluate(test_x, test_y)


def main():
    pass


if __name__ == '__main__':
    main()
