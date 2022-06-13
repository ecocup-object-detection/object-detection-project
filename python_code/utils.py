import os
import pandas as pd
import random


def get_IoU(box1, box2):
    """Return the Intersection over Union between 2 boxes

    Args:
        box1 (tuple): tuple containing x and y coordinates + width and height of the box 1
        box2 (tuple): tuple containing x and y coordinates + width and height of the box 2
    """

    x1_1 = box1[0]
    y1_1 = box1[1]
    x1_2 = x1_1 + box1[2]
    y1_2 = y1_1 + box1[3]

    x2_1 = box2[0]
    y2_1 = box2[1]
    x2_2 = x2_1 + box2[2]
    y2_2 = y2_1 + box2[3]

    x_1 = max(x1_1, x2_1)
    x_2 = min(x1_2, x2_2)
    y_1 = max(y1_1, y2_1)
    y_2 = min(y1_2, y2_2)

    intersection_area = max(0, x_2 - x_1) * max(0, y_2 - y_1)
    box1_area = (x1_2 - x1_1) * (y1_2 - y1_1)
    box2_area = (x2_2 - x2_1) * (y2_2 - y2_1)
    iou = intersection_area/(box1_area + box2_area - intersection_area)

    return iou


def load_images_names(pos=True, neg=True):
    """Load all images names in order to create train and
    validation data

    Args:
        pos (bool, optional): Include positive images. Defaults to True.
        neg (bool, optional): Include negative images. Defaults to True.
    """
    if pos:
        images_pos = os.listdir("../dataset/train/images/pos/")
        list_image_pos = []
        for image in images_pos:
            list_image_pos.append(f"../dataset/train/images/pos/{image}")
    if neg:
        images_neg = os.listdir("../dataset/train/images/neg/")
        list_image_neg = []
        for image in images_neg:
            list_image_neg.append(f"../dataset/train/images/neg/{image}")
    images = list_image_neg + list_image_pos
    random.Random(20).shuffle(images)

    df = pd.DataFrame(images, columns=["Image"])

    return df
