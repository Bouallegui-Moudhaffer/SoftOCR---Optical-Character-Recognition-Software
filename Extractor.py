# Imports:
import cv2
import glob
import re
import pytesseract
import os
from natsort import os_sorted
import fileinput

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'


def extract_names():
    # Adding custom options
    img_list = []
    for img in glob.glob(r"name_results\*.png"):
        img_list.append(img)
    # Natural sorting method
    img_list = os_sorted(img_list)
    # print(img)*
    # For each image call extract_name() defined bellow function on it
    for img in img_list:
        extract_name(img)
    # Read through temp.txt file which contains the first extraction output by Tesseract
    with open("temp.txt", "r") as f:
        names = f.readlines()
    f.close()
    # Text Post-Processing
    names = [n.replace("\n", "") for n in names]
    # Joining every two lines to get every First and Last name in the same line seperated by a comma
    names = [",".join(names[i:i + 2]) for i in range(0, len(names), 2)]
    with open(r"saved_names.txt", "w") as fo:
        fo.write("\n".join(names))
    fo.close()
    # print(names)


def extract_name(img):
    words = []
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)
    text = text.replace("\n", " ")
    # print(text)
    # Text Post-Processing
    x = re.sub(r'[^ \sA-Z/]+', '', text)
    # print(x)
    words.append(x)
    print(x)
    # Leaving only uppercase strings
    all_caps = list([s.strip() for s in words if s == s.upper() and s != 'NOM' and s != 'PRENOM'])
    # Removing blank lines
    no_blank = list([string for string in all_caps if string != ""])
    # Writing to temp.txt
    with open('temp.txt', 'a+') as filehandle:
        # print(no_blank)
        for listitem in no_blank:
            filehandle.write(f'{listitem}\n')
    filehandle.close()
