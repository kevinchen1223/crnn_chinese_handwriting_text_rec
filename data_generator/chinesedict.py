# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 21:54:38 2019

@author: Nuo Xu
"""
import os
import random
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import imageio

char_dictionary = np.load('charDict.npy').item()
#%%
strchar = ""
for item in char_dictionary.keys():
    strchar += str(item.decode('utf-8')