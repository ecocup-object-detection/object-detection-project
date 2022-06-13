import cv2
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from utils import get_IoU, load_images_names

df = load_images_names(pos=True, neg=True)
X_train, X_val = train_test_split(
    df["Image"], random_state=15)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

train_images = []
train_labels = []
nb_pos = 0
nb_viewed = 0
for i in range(len(X_train)):
    image = list(X_train)[i]
    img = cv2.imread(image)
    picture_name = image[28:-4]
    try:
        if "pos" in picture_name:
            ecocup_locations = []
            labels = pd.read_csv(
                f"../dataset/train/labels_csv/{picture_name}.csv", header=None)
            labels.columns = ["y", "x", "h", "w", "class"]
            nb_viewed += 1
            print(nb_viewed, " File analyzed: ", picture_name)
            for label in labels.iterrows():
                x = labels["x"][0]
                y = labels["y"][0]
                w = labels["w"][0]
                h = labels["h"][0]
                ecocup_locations.append(
                    {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h
                    }
                )
            ss.setBaseImage(img)
            ss.switchToSelectiveSearchFast()
            ssresults = ss.process()
            imout = img.copy()
            for e, result in enumerate(ssresults):
                for location in ecocup_locations:
                    x = location["x"]
                    y = location["y"]
                    w = location["w"]
                    h = location["h"]
                    x_result, y_result, w_result, h_result = result
                    iou = get_IoU(
                        (x, y, w, h),
                        (x_result, y_result, w_result, h_result)
                    )
                    if h_result > 0.05*imout.shape[0] and w_result > 0.05*imout.shape[1]:
                        if iou > 0.7:
                            train_image = imout[y_result:y_result +
                                                h_result, x_result:x_result+h_result]
                            resized = cv2.resize(
                                train_image, (224, 224), interpolation=cv2.INTER_AREA)
                            train_images.append(resized)
                            train_labels.append(1)
                            nb_pos += 1
                        elif iou < 0.2:
                            add = np.random.binomial(n=1, p=0.007)
                            if add == 1:
                                train_image = imout[y_result:y_result +
                                                    h_result, x_result:x_result+h_result]
                                resized = cv2.resize(
                                    train_image, (224, 224), interpolation=cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(0)
        elif "neg" in picture_name:
            nb_viewed += 1
            print(nb_viewed, " File analyzed: ", picture_name)
            ss.setBaseImage(img)
            ss.switchToSelectiveSearchFast()
            ssresults = ss.process()
            imout = img.copy()
            for e, result in enumerate(ssresults):
                x_result, y_result, w_result, h_result = result
                if h_result > 0.05*imout.shape[0] and w_result > 0.05*imout.shape[1]:
                    add = np.random.binomial(n=1, p=0.007)
                    if add == 1:
                        train_image = imout[y_result:y_result +
                                            h_result, x_result:x_result+h_result]
                        resized = cv2.resize(
                            train_image, (224, 224), interpolation=cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(0)

        print("Negative label: ", len(train_labels)-nb_pos,
              " --------------- Postive label: ", nb_pos)
    except Exception as exception:
        print(exception)
        print("error on file: ", picture_name)

np.save("../train_data/train_images_full", np.array(train_images))
np.save("../train_data/train_labels_full", np.array(train_labels))
