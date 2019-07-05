# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import imageio
import scipy.misc
import json
#%%
train_dir = "C:\\Users\\Nuo Xu\\Desktop\\WORK\\Intern\\Solution1\\train"
basewidth = 60
TextLength = random.randint(10,20)
numExamples = 200
example_path =  "C:\\Users\\Nuo Xu\\Desktop\\WORK\\Intern\\Solution1\\"


# %%
def resize_text(img, basewidth):
    height, width, depth  = img.shape
    imgScale  = basewidth / height 
    newX,newY = img.shape[1]*imgScale, img.shape[0]*imgScale
    newimg = cv2.resize(img, (int(np.round(newX)), int(np.round(newY))))
    return newimg
# %%
def get_image_path_and_labels(dir):
    img_path = []
    for root, dir, files in os.walk(dir):
        img_path += [os.path.join(root, f) for f in files]
    # Shuffle the data to avoid overfit
    random.shuffle(img_path)
    # Because I created folders corresponding to each character, so the folder name is actual label
    labels = [int(name.split(os.sep)[len(name.split(os.sep)) - 2]) for name in img_path]
    return img_path, labels
img_path, labels = get_image_path_and_labels(train_dir)
# %%
text_labels = dict()
char_dictionary = np.load('charDict.npy').item()
file = open("C:\\Users\\Nuo Xu\\Desktop\\WORK\\Intern\\Solution1\\test_width.txt", "w")
blank = np.ones((basewidth, 15, 3), dtype=np.uint8) * 255
for i in range(numExamples):
    print(i)
    TextLength = random.randint(10,20)
    random_lib = random.sample(range(0, len(labels)), TextLength)
    
    img = cv2.imread(img_path[random_lib[0]])
    text_labels.update({i:[]})
    charValue =  labels[random_lib[0]]
    ChineseChar = list(char_dictionary.keys())[charValue]
    text_labels[i].append(ChineseChar)
    result = resize_text(img, basewidth)
    result = np.concatenate((result, blank), axis = 1)
    for j in range(1, TextLength):
        img = cv2.imread(img_path[random_lib[j]])
        newimg = resize_text(img, basewidth)
        charValue = labels[random_lib[j]]
        ChineseChar = list(char_dictionary.keys())[charValue]
        text_labels[i].append(ChineseChar)
        result = np.concatenate((result, newimg),axis = 1)
        result = np.concatenate((result, blank), axis = 1)
    if not os.path.exists(os.path.join(example_path, 'test_width')):
        os.mkdir(os.path.join(example_path, 'test_width'))
    tmp = os.path.join(example_path,'test_width')
    imageio.imwrite(os.path.join(tmp,'%05d.jpg' % i), result)
    file.write('%05d.jpg ' % i  )
    for item in text_labels[i]:
        file.write("%s" % item)
    file.write('\n'  )
    #file.writelines(['%05d.jpg' % i , ' ', text_labels[i]])
    
file.close()