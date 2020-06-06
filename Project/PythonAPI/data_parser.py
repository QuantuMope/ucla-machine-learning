from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import pickle


"""
    Download COCO 2014 train/val annotations: http://cocodataset.org/#download
    Make sure to change the data directory below to wherever you store the annotation data.
    
    Follow these steps to install pycocotools.coco: https://github.com/cocodataset/cocoapi
    You can simply cd into PythonAPI directory and run "make"
    
    Probably don't need to look at this unless you want to alter the data accessed.
    COCO api: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
"""

dataDir = '../../..'
dataType = 'train2014'
annotations = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
captions = '{}/annotations/captions_{}.json'.format(dataDir, dataType)

coco_anns = COCO(annotations)
coco_caps = COCO(captions)

male_biased = ['truck', 'motorcycle', 'tie', 'backpack', 'sports ball']
# some of these are actually male biased due to lack of female biased terms
female_biased = ['handbag', 'fork', 'knife', 'spoon', 'cell phone', 'teddy bear']

bias_terms = female_biased + male_biased
training_set = {}  # dict containing imgIds: labels
male_count = 0
female_count = 0

for term in bias_terms:
    catIds = coco_anns.getCatIds(catNms=['person', term])
    imgIds = coco_anns.getImgIds(catIds=catIds)
    for imgId in imgIds:
        capId = coco_caps.getAnnIds(imgIds=imgId)
        captions = coco_caps.loadAnns(capId)
        male_detected = False
        female_detected = False
        male_and_female = False
        for cap in captions:
            cap_string = cap['caption']
            if ' man' in cap_string:  # add space so woman is not counted in man count
                male_detected = True
            if 'woman' in cap_string:
                female_detected = True
            if male_detected and female_detected:
                male_and_female = True
                break
        if male_and_female: continue
        img_id = captions[0]['image_id']
        # if image has already been added previously, simply add the additional label
        if img_id in training_set.keys():
            training_set[img_id].append(term)
            continue
        # if image has been added for first time, manually add the {man, woman} label
        if male_detected:
            training_set[img_id] = ['man']
            training_set[img_id].append(term)
            male_count += 1
        elif female_detected:
            training_set[img_id] = ['woman']
            training_set[img_id].append(term)
            female_count += 1

print('{} total male only images'.format(male_count))
print('{} total female only images'.format(female_count))
print('{} total training examples'.format(len(training_set)))
# Calculate bias in training set, first index is male, second index is female
label_counts = {term: [0, 0] for term in bias_terms}
train_imgIds = list(training_set.keys())
for tr_img in train_imgIds:
    labels = training_set[tr_img]
    gender = labels[0]
    for label in labels[1:]:
        if gender is 'man':
            label_counts[label][0] += 1
        elif gender is 'woman':
            label_counts[label][1] += 1

"""
    All selected examples for training set are in training_set.
    The keys in the training_set dict contain all image ids.
    These image ids are used by the loadImgs method in coco_anns
    to get the image information. This information contains a 'coco_url'
    that can be used to download the image. You can uncomment the code
    below and test it. I suggest once you download all the images to
    serialize it for your guys' usage. COCO API has a download method
    that may come in handy.
"""

IMG_DATA_DIRECTORY = './testing_img_data'

loaded_imgs = coco_anns.loadImgs(training_set.keys())

print('Starting image data download')
all_test_img_data = []
for i, test_image in enumerate(loaded_imgs):
    all_test_img_data.append(io.imread(test_image['coco_url']))
    print('Downloading image {}...'.format(i + 1))
print('Finished image data download')

with open(IMG_DATA_DIRECTORY, 'wb') as f:
    pickle.dump(all_test_img_data, f)

"""
    Once you serialize once, you can use the following code in a separate program.
    
    with open(IMG_DATA_DIRECTORY, 'rb') as f:
        all_test_img_data = pickle.load(f)
"""


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0.4, 1.0)
ax.set_ylim(0.4, 1.0)
plt.xlabel('Training Set Gender Ratio')
plt.ylabel('Predicted Gender Ratio')
plt.title('Bias analysis on MS-COCO MLC')
ax.plot([0, 1], [0, 1], c='b')

# Calculate ratios from label counts and plot
for term in bias_terms:
    counts = label_counts[term]
    ratio = counts[0] / sum(counts)
    plt.scatter(ratio, ratio)
    plt.text(ratio, ratio, term)

plt.show()

