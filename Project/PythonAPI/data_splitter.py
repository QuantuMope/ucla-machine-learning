import numpy as np
import pickle
from sklearn.model_selection import train_test_split


label_keys = {'gender': 0, 'truck': 1, 'motorcycle': 2, 'tie': 3, 'backpack': 4, 'sports ball': 5,
              'handbag': 6, 'fork': 7, 'knife': 8, 'spoon': 9, 'cell phone': 10, 'teddy bear': 11}


def one_hot_encode(y):
    train_y = np.zeros((len(y), len(label_keys)))
    for i, labels in enumerate(y):
        train_y[i, 0] = 1 if labels[0] == 'man' else 0
        for label in labels[1:]:
            train_y[i, label_keys[label]] = 1
    return train_y


tr_data_file = 'feature_embedding.pkl'
X = np.load(tr_data_file, allow_pickle=True)

tr_label_file = 'testing_img_labels.pkl'
with open(tr_label_file, 'rb') as f:
    raw_y = pickle.load(f)

y = one_hot_encode(raw_y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

np.save('data/X_train', X_train)
np.save('data/X_test', X_test)
np.save('data/y_train', y_train)
np.save('data/y_test', y_test)
