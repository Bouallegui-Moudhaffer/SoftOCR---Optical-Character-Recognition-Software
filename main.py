from Convertor import convert
from Preprocessor import correct_skew
from Locator import locate_marks, locate_names
from Extractor import extract_names, extract_name
from keras.models import model_from_json
import os, glob
from Cnn import predict_digits, cnn_model
import numpy as np
import cv2

# convert('best (0).pdf')
#
# correct_skew(r"C:\Users\lenovo\PycharmProjects\SoftOCR_PFE\Images")
#
# locate_names('Corrected_images/corrected_img3.png')
#
# extract_names()
#
#
# # Create, train model and save it
# cnn_model()
# locate_marks(r"Corrected_images\corrected_img3.png")
# load json and create model
json_file = open('H5_FILE.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("H5_FILE.h5")
print("Loaded model from disk")
prediction_list = []
for img in glob.glob("marks_results/*.png"):
    prediction_list.append(predict_digits(loaded_model, img))

# Open for read/write with file initially at beginning
with open("saved_names.txt", "r+") as f:
    # Make new lines
    newlines = ['{},{}\n'.format(old.rstrip('\n'), toadd) for old, toadd in zip(f, prediction_list)]
    f.seek(0)  # Go back to beginning for writing
    f.writelines(newlines)  # Write new lines over old
    f.truncate()  # Not needed here, but good habit if file might shrink from change

