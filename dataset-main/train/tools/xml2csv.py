# -*- coding: utf-8 -*-
"""
Created on Mon May 16 23:37:52 2022

@author: xuphilip
"""
import os
import numpy as np
from lxml import etree

labels = os.listdir('labels')
for lbl in labels:
    root = etree.parse(os.path.join('labels', lbl))
    objs = root.findall("object")
    bndboxes = np.zeros((len(objs), 5))
    for idx, obj in enumerate(objs):
        d = int(obj.find("difficult").text)
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        i = ymin
        j = xmin
        h = ymax - ymin
        w = xmax - xmin
        bndboxes[idx, :] = np.array([i, j, h, w, d])
    np.savetxt(os.path.join('labels_csv', lbl[:-4] + '.csv'), bndboxes, fmt='%i', delimiter=",")
