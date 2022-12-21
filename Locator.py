import glob
import os
import re
import shutil
from keras.models import model_from_json
import cv2
import numpy as np
from Cnn import predict_digits


# Locate marks_results
def locate_marks(mark_img):
    i = 1
    table_image_contour = cv2.imread(mark_img, 0)
    table_image = cv2.imread(mark_img)

    ret, thresh_value = cv2.threshold(table_image_contour, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    dilated_value = cv2.dilate(thresh_value, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda x: get_contour_precedence(x, table_image.shape[1]))
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # bounding the
        if x > 2029 and y > 770 and w > 150:
            table_image = cv2.rectangle(table_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
            roi = table_image[y: y + h, x: x + w]
            cv2.imwrite(rf'marks_results\mark{i}.png', roi)
            i += 1
    # cv2.imwrite(rf'temp\out2.png', table_image)
    # load json and create model
    # json_file = open('H5_FILE.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("H5_FILE.h5")
    # print("Loaded model from disk")
    # prediction = []
    # for img in glob.glob("marks_results/*.png"):
    #     prediction.append(predict_digits(loaded_model, img))
    # # Code to add marks_results later on
    # print(prediction)
    # # for element in prediction:
    # with open("saved_names.txt", 'r') as f:
    #     file_lines = [','.join([x.strip(), prediction[i], '\n']) for x in f.readlines()]
    # f.close()
    # with open('saved_names.txt', 'w') as f:
    #     f.writelines(file_lines)
    #     f.close()
    # return prediction


def locate_names(img):
    # Checking if directory exists and clearing it of previously stored files
    if not os.path.exists(rf'C:\Users\lenovo\PycharmProjects\SoftOCR_PFE\name_results'):
        os.mkdir(rf'C:\Users\lenovo\PycharmProjects\SoftOCR_PFE\name_results')
    else:
        for filename in os.listdir(rf'C:\Users\lenovo\PycharmProjects\SoftOCR_PFE\name_results'):
            file_path = os.path.join(rf'C:\Users\lenovo\PycharmProjects\SoftOCR_PFE\name_results', filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s: %s' % (file_path, e))
    # Load images
    table_image_contour = cv2.imread(img, 0)
    table_image = cv2.imread(img)
    # Threshold the image
    _, thresh_value = cv2.threshold(table_image_contour, 180, 255, cv2.THRESH_BINARY_INV)
    # Apply dilatation on image using kernel
    kernel = np.ones((5, 5), np.uint8)
    dilated_value = cv2.dilate(thresh_value, kernel, iterations=1)
    # Finding contours in the dilated image
    contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours based on positional precedence (left to right - top to bottom)
    contours.sort(key=lambda x: get_contour_precedence(x, table_image.shape[1]))
    i, j = 0, 0
    for cnt in contours:
        # Getting (x, y) values for the first pixel in a contour and its width and height
        x, y, w, h = cv2.boundingRect(cnt)
        # Filtering through contours using their x, y positions and their width as heuristic
        if 1390 > x > 478 and 1000 > w > 400 and y > 770:
            # Drawing bounding rectangles on each contour
            image = cv2.rectangle(table_image, (x, y), (x + w, y + h), (0, 0, 255), 4)
            # Cropping and saving each region of interest ie: a single contour using its x, y and width, height values
            roi = table_image_contour[y: y + h, x: x + w]
            cv2.imwrite(rf'name_results\name{i}.png', roi)
            i += 1
    # cv2.imwrite(rf'name_result\out{j}.png', image)

# Function for getting precedence for contours.


def get_contour_precedence(contour, cols):
    origin = cv2.boundingRect(contour)
    return origin[1] * cols + origin[0]
