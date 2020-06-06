import numpy as np
import skimage.io as io
from skimage.transform import resize
import pickle

dataDir = '../../..'
image_dir = '{}/images/'.format(dataDir)

IMG_DATA_DIRECTORY = 'testing_img_data.pkl'

with open(IMG_DATA_DIRECTORY, 'rb') as f:
    all_test_img_data = pickle.load(f)

num_training_samples = len(all_test_img_data)
training_data = np.zeros((num_training_samples, 3, 224, 224), dtype=np.float64)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
for i, img_name in enumerate(all_test_img_data):
    data = io.imread(image_dir + img_name)
    resized_data = resize(data, (224, 224, 3))  # auto scaling [0, 1]
    reordered_data = np.rollaxis(resized_data, 2, 0)  # reorder shape to comply with pytorch

    preprocessed_data = np.zeros((3, 224, 224), dtype=np.float64)
    # Normalize data according to mean std given by pytorch
    preprocessed_data[0] += (reordered_data[0] - mean[0]) / std[0]
    preprocessed_data[1] += (reordered_data[1] - mean[1]) / std[1]
    preprocessed_data[2] += (reordered_data[2] - mean[2]) / std[2]

    training_data[i] += preprocessed_data
    print('Preprocessed image {}...'.format(i + 1))

PREPROCESSED_IMG_DIR = 'preprocessed_testing_img_data.pkl'
with open(PREPROCESSED_IMG_DIR, 'wb') as f:
    np.save(f, training_data)

print('break')


