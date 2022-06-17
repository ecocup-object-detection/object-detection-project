"""This file is used to generate the results on the test
images with the VGEcocup model
"""
import cv2
import pandas as pd
import tensorflow as tf

from utils import (
    load_test_images_names,
    get_image,
    get_selective_search_boxes,
    get_vgecocup_output,
    non_max_suppression,
    flatten
)

df = load_test_images_names(test_folder_filepath="../dataset/test")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
filepathes_test = list(df["Image"])
vgecocup = tf.keras.models.load_model("../models/vgecocup_version2")

final_output = []
for filepath in filepathes_test:
    img, picture_name = get_image(filepath)
    print("Picture: ", picture_name, ".jpg")
    print("Shape: (", img.shape[0], ", ", img.shape[1], ")")
    ssresults = get_selective_search_boxes(img, ss)
    model_output = get_vgecocup_output(ssresults, img, vgecocup)
    model_output_nms = non_max_suppression(
        model_output, 0.1, picture_name, img)
    final_output.append(model_output_nms)

final_output_flattened = flatten(final_output)
df_output = pd.DataFrame(final_output_flattened)
df_output = df_output[["filename", "y", "x", "h", "w", "score"]]
df_output.to_csv("../vgecocup_results_version2.csv", index=False, header=False)
