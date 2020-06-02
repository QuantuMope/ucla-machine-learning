from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab

dataDir = '../../..'
dataType = 'train2014'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
names = [cat['name'] for cat in cats]
print(len(names))
print('COCO categories: \n{}\n'.format(' '.join(names)))
