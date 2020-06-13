import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
from tensorflow.keras.optimizers import Adam, SGD


class BaselineNN:
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

        outputs = Dense(12, activation='sigmoid')(hd)

        model = Model(inputs, outputs)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

    @staticmethod
    def _dense_batch_layer(num_units, prev_layer):
        hd = Dense(num_units,
                   kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                   kernel_initializer='he_uniform')(prev_layer)
        hd = BatchNormalization()(hd)
        hd = ReLU()(hd)
        return hd

    def train_model(self, train_x, train_y, batch_size, epochs):
        self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.05, shuffle=True)
        self.model.save_weights(SAVE_DIR)

    def evaluate(self, test_x, test_y):
        pred_y = self.predict(test_x)
        pred_y[pred_y > 0.5] = 1.
        pred_y[pred_y < 0.5] = 0.
        label_diff = pred_y - test_y
        partial_accuracy = 1 - np.mean(np.abs(label_diff))
        print('Partial match accuracy: {}'.format(partial_accuracy))

        num_instances = pred_y.shape[0]
        all_zeros = np.zeros(pred_y.shape[1])
        exact_matches = np.count_nonzero((label_diff == all_zeros).all(axis=1))
        exact_math_ratio = exact_matches / num_instances
        print('Exact match accuracy: {}'.format(exact_math_ratio))

        actual_label_loc = np.where(test_y == 1)
        actu_label = test_y[actual_label_loc]
        pred_label = pred_y[actual_label_loc]
        precision = pred_label.sum() / actu_label.sum()
        print('Precision score: {}'.format(precision))

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

        hfont = {'fontname': 'Times New Roman'}
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.0)
        plt.xlabel('Data Set Gender Ratio', size=12, **hfont)
        plt.ylabel('Predicted Gender Ratio', size=12, **hfont)
        plt.title('Baseline NN Gender Bias analysis on MS-COCO MLC', size=14, **hfont)
        ax.plot([0, 1], [0, 1], c='b')
        ratios = []
        for i in range(11):
            pred_counts = pred_label_counts[i]
            true_counts = true_label_counts[i]
            pred_ratio = pred_counts[0] / sum(pred_counts)
            true_ratio = true_counts[0] / sum(true_counts)
            ratios.append((pred_ratio, true_ratio))
            plt.scatter(true_ratio, pred_ratio)
            plt.text(true_ratio, pred_ratio, self.label_keys[i], size=12, **hfont)

        # Calculate average bias amplification.
        bias_amplication = 0
        for pred_ratio, true_ratio in ratios:
            bias_amplication += abs(pred_ratio - true_ratio)

        bias_amplication /= len(ratios)
        print('Average bias amplication: {}'.format(bias_amplication))

        plt.show()


LOAD_DIR = './model_weights/model_weights'  # 50 epochs
SAVE_DIR = LOAD_DIR


def main():
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')

    baseline_nn = BaselineNN(input_size=1024,
                             # optimizer=SGD(lr=0.0001, momentum=0.9),
                             optimizer=Adam(lr=0.00003),
                             load_weights=True)

    # baseline_nn.train_model(X_train, y_train, batch_size=32, epochs=25)

    baseline_nn.evaluate(X_train, y_train)
    baseline_nn.evaluate(X_test, y_test)

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    baseline_nn.calculate_gender_bias(X, y)

    print('break')


if __name__ == '__main__':
    main()
