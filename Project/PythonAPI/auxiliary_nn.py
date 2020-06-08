import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.math import log, reduce_sum
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.optimizers import Adam, SGD


class AuxiliaryNN:
    def __init__(self, input_size, optimizer, load_weights=False):

        self.input_size = input_size  # input size of feature space output from Resnet
        self.optimizer = optimizer

        self.model = self._build_network()
        if load_weights:
            self.model.load_weights(LOAD_DIR)

        self.label_keys = {0: 'truck', 1: 'motorcycle', 2: 'tie', 3: 'backpack', 4: 'sports ball',
                           5: 'handbag', 6: 'fork', 7: 'knife', 8: 'spoon', 9: 'cell phone',
                           10: 'teddy bear'}

    def _build_network(self):
        inputs = Input(shape=self.input_size)
        hd = self._dense_batch_layer(1024, inputs)
        hd = self._dense_batch_layer(512, hd)
        hd = self._dense_batch_layer(256, hd)
        hd = self._dense_batch_layer(256, hd)
        hd = self._dense_batch_layer(128, hd)
        hd = self._dense_batch_layer(128, hd)
        hd = self._dense_batch_layer(64, hd)
        hd = self._dense_batch_layer(64, hd)

        classification_output = Dense(12, activation='sigmoid')(hd)
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
        batch_size = true_y.shape[0]
        class_true_y = true_y[:, :-1]
        class_pred_y = pred_y[:, :-1]
        regre_true_y = true_y[:, -1]
        regre_pred_y = pred_y[:, -1]

        # cross-entropy loss
        main_loss = -reduce_sum(class_true_y * log(class_pred_y))
        # mean absolute loss
        auxiliary_loss = reduce_sum(tf.abs(regre_true_y - regre_pred_y)) / batch_size
        total_loss = main_loss + LAMBDA_ * auxiliary_loss
        return total_loss

    @staticmethod
    def _dense_batch_layer(num_units, prev_layer):
        hd = Dense(num_units,
                   kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                   kernel_initializer='he_uniform')(prev_layer)
        hd = BatchNormalization()(hd)
        hd = ReLU()(hd)
        return hd

    def train_model(self, train_x, train_y, batch_size, epochs):
        self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.1, shuffle=True)

    def evaluate(self, test_x, test_y):
        print(self.model.evaluate(test_x, test_y))

    def predict(self, x):
        return self.model.predict(x)

    def calculate_gender_bias(self, x, y):
        pred_y = self.predict(x)
        pred_label_counts = {i: [0, 0] for i in range(11)}
        for entry in pred_y:
            pred_gender = 'man' if entry[0] > 0.5 else 'woman'
            ones = np.where(entry[1:] > 0.5)[0]
            for one in ones:
                if pred_gender == 'man':
                    pred_label_counts[one][0] += 1
                else:
                    pred_label_counts[one][1] += 1

        true_label_counts = {i: [0, 0] for i in range(11)}
        for entry in y:
            pred_gender = 'man' if entry[0] > 0.5 else 'woman'
            ones = np.where(entry[1:] > 0.5)[0]
            for one in ones:
                if pred_gender == 'man':
                    true_label_counts[one][0] += 1
                else:
                    true_label_counts[one][1] += 1

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.0)
        plt.xlabel('Training Set Gender Ratio')
        plt.ylabel('Predicted Gender Ratio')
        plt.title('Auxiliary NN Gender Bias analysis on MS-COCO MLC')
        ax.plot([0, 1], [0, 1], c='b')
        for i in range(11):
            pred_counts = pred_label_counts[i]
            true_counts = true_label_counts[i]
            pred_ratio = pred_counts[0] / sum(pred_counts)
            true_ratio = true_counts[0] / sum(true_counts)
            plt.scatter(true_ratio, pred_ratio)
            plt.text(true_ratio, pred_ratio, self.label_keys[i])

        plt.show()


LOAD_DIR = './aux_model_weights'
SAVE_DIR = LOAD_DIR
LAMBDA_ = 0.10  # auxiliary loss coefficient


def main():
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')

    auxiliary_nn = AuxiliaryNN(input_size=1024,
                               optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                               # optimizer=tf.keras.optimizers.Adam(lr=0.00003),
                               load_weights=False)


    print('break')


if __name__ == '__main__':
    main()
