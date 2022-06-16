import cv2
import numpy as np
import os
import pandas as pd
import random

from typing import List, Dict, Tuple


def get_IoU(box1: Tuple, box2: Tuple) -> float:
    """Return the Intersection over Union between 2 boxes

    Args:
        box1 (tuple): tuple containing x and y coordinates + width and height of the box 1
        box2 (tuple): tuple containing x and y coordinates + width and height of the box 2

    Returns:
        iou (float): Intersection over Union of the 2 boxes
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


def load_images_names(pos=True, neg=True) -> pd.DataFrame:
    """Load all images names in order to create train and
    validation data

    Args:
        pos (bool, optional): Include positive images. Defaults to True.
        neg (bool, optional): Include negative images. Defaults to True.

    Returns:
        df (pd.DataFrame): pandas dataframe with all filepathes of positive
            and/or negatives images
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


def non_max_suppression(output_model: List[Dict], threshold: float, picture_name: str) -> List:
    """Apply Non Max Suppression on a list of boxes in order
    to keep just the better bounding box as a detection 

    Args:
        output_model (List[Dict]): list of dictionnaries containing
            boxes coordinates (x, y, w, h) and score of detection
        threshold (float): threshold of IoU: if the box has an IoU
            bigger than the threshold with the better bounding box,
            this box will be removed.
        picture_name (str) 

    Returns:
        List: _description_
    """
    boxes_output = []
    boxes = output_model.copy()
    while len(boxes) >= 1:
        boxes_used = sorted(boxes, key=lambda x: x["score"], reverse=True)
        best_prediction = boxes_used[0]
        prediction = {}
        prediction["filename"] = f"{picture_name}.jpg"
        prediction.update(best_prediction)
        boxes_output.append(prediction)
        boxes.remove(best_prediction)
        boxes_used.remove(best_prediction)
        for i in range(len(boxes_used)):
            box = boxes_used[i]
            iou = get_IoU(
                (
                    best_prediction["x"],
                    best_prediction["y"],
                    best_prediction["w"],
                    best_prediction["h"]
                ),
                (
                    box["x"],
                    box["y"],
                    box["w"],
                    box["h"]
                )
            )
            if iou > threshold:
                boxes.remove(box)
    return boxes_output


def get_ecocup_locations_from_labelized_data(picture_name: str) -> List[Dict]:
    """Get all the labelized ecocup location from a positive train image

    Args:
        picture_name (str): Name of the picture, utils to load the
            correspondant csv file

    Returns:
        List[Dict]: all the boxes coordinates of the ecocups in the
            image
    """
    ecocup_locations = []
    labels = pd.read_csv(
        f"../dataset/train/labels_csv/{picture_name}.csv", header=None)
    labels.columns = ["y", "x", "h", "w", "class"]
    for label in labels.iterrows():
        x = label["x"][0]
        y = label["y"][0]
        w = label["w"][0]
        h = label["h"][0]
        ecocup_locations.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            }
        )

    return ecocup_locations


def load_test_images_names(test_folder_filepath: str) -> pd.DataFrame:
    """Load all filepath of test images

    Args:
        filepath (str): Test folder filepath (absolute or relative)

    Returns:
        df (pd.DataFrame): pandas dataframe containing all filepathes
            of test images
    """
    images = os.listdir(test_folder_filepath)
    list_image = []
    for image in images:
        list_image.append(f"{test_folder_filepath}/{image}")
    df = pd.DataFrame(list_image, columns=["Image"])

    return df


def get_image(filepath: str) -> Tuple[np.array, str]:
    """Get the image containing at the filepath

    Args:
        filepath (str): filepath of the image

    Returns:
        Tuple[np.array, str]: image and image name
    """
    img = cv2.imread(filepath)
    picture_name = filepath[16:-4]
    return img, picture_name


def get_selective_search_boxes(image, ss):
    """_summary_

    Args:
        image (np.array): image data
        ss (cv2 object): Selective Segmentation declaration

    Returns:
        ssresults: All the Region of interest returned by the
            Selective Search Segmentation
    """
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()

    return ssresults


def get_vgecocup_output(ssresults, img, classification_model):
    """Get the result of the detection algorithm

    Args:
        ssresults: results of Selective Search Segmentation
        img (str): image name
        classification_model: machine/deep learning model

    Returns:
        model_output: All the bounding boxes of ecocup detected by the
            algorithm
    """
    model_output = []
    imout = img.copy()
    for e, boxes in enumerate(ssresults):
        if e < 3000 or e == len(ssresults):
            x_result, y_result, w_result, h_result = boxes
            if h_result > 0.1*imout.shape[0] and w_result > 0.1*imout.shape[1]:
                test_image = imout[y_result:y_result +
                                   h_result, x_result:x_result+h_result]
                resized = cv2.resize(test_image, (224, 224),
                                     interpolation=cv2.INTER_AREA)
                input_model = np.expand_dims(resized, axis=0)
                classification_model_result = classification_model.predict(
                    input_model)
                if classification_model_result[0][1] > 0.9:
                    model_output.append(
                        {
                            "x": x_result,
                            "y": y_result,
                            "w": w_result,
                            "h": h_result,
                            "score": classification_model_result[0][1]
                        }
                    )

    return model_output


def flatten(list_of_list):
    """Flatten a list of list of dict

    Args:
        list_of_list (List[List[Dict]]): list of list of dict to flatten

    Returns:
        (List[Dict]): input list flattened
    """
    return [dict_obtain for list_of_dict in list_of_list for dict_obtain in list_of_dict]
