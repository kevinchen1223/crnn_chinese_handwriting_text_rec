# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import imageio
import scipy.misc
import json
#%%
# Define Parameters
train_dir           = "C:\\Users\\Nuo Xu\\Desktop\\WORK\\Intern\\Solution1\\train"
basewidth           = 60
TextLength          = random.randint(10,20)
numExamples         = 200000
example_path        =  "C:\\Users\\Nuo Xu\\Desktop\\WORK\\Intern\\Solution1\\"
Data_Augumentation  = True
Salt_and_Pepper     = True
Affine_Flag         = True
Drop_Flag           = True
num_Salt_and_Pepper = 300
num_Obsculation_box = 15

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

# Read the char dictionaries fron npy
text_labels = dict()
char_dictionary = np.load('charDict.npy').item()
file = open("C:\\Users\\Nuo Xu\\Desktop\\WORK\\Intern\\Solution1\\train_noise.txt", "w")
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
        if Data_Augumentation and Affine_Flag and np.random.randint(0,10) > 5:
            h, w, _ = img.shape
            warp_a = np.random.uniform(0,0.6)
            M_crop_Eason = np.array([
                [1, warp_a, 0],
                [0, 1, 0]
                ], dtype=np.float32)
            img = cv2.warpAffine(img, M_crop_Eason, (int(np.round(w + warp_a * h )) , h), borderValue=(255,255,255))
            
        newimg = resize_text(img, basewidth)
        charValue = labels[random_lib[j]]
        ChineseChar = list(char_dictionary.keys())[charValue]
        text_labels[i].append(ChineseChar)
        result = np.concatenate((result, newimg),axis = 1)
        result = np.concatenate((result, blank), axis = 1)
    
    if Data_Augumentation and Drop_Flag and np.random.randint(10) > 5:
        box_h, box_w = 4, 6
        rows, cols, channel = result.shape
        x = np.random.randint(rows-box_h, size = num_Obsculation_box)
        y = np.random.randint(cols-box_w, size = num_Obsculation_box)
        for (x1, y1) in zip(x,y):
            result = cv2.rectangle(result, (y1, x1), (y1+box_w, x1+box_h), (0,0,0), cv2.FILLED)
    

        
    # Create a dir to save images
    if not os.path.exists(os.path.join(example_path, 'train_noise')):
        os.mkdir(os.path.join(example_path, 'train_noise'))
    
    # Add Salt and Peper Noises
    if Salt_and_Pepper and Data_Augumentation:
        rows, cols, channel = result.shape
        x = np.random.randint(rows, size = num_Salt_and_Pepper)
        y = np.random.randint(cols, size = num_Salt_and_Pepper)
        half = num_Salt_and_Pepper // 2
        result[x[:half],y[:half],:] = 255
        result[x[half:],y[half:],:] = 0
    tmp = os.path.join(example_path,'train_noise')
    imageio.imwrite(os.path.join(tmp,'%05d.jpg' % i), result)
    file.write('%05d.jpg ' % i  )
    for item in text_labels[i]:
        file.write("%s" % item)
    file.write('\n'  )
    
    
file.close()