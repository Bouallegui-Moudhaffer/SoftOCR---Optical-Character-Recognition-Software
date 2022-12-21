import os
from skimage.filters import median
from skimage.morphology import disk
from wand.image import Image
import glob


# De-skew raw images
def correct_skew(folder):
    i = 0
    filenames = os.listdir(folder)
    filenames.sort()
    for filename in filenames:
        print(filename)
        with Image(filename=rf'{folder}\{filename}') as img:
            img.deskew(0.4 * img.quantum_range)
            img.save(filename=rf'Corrected_images\corrected_img{i}.png')
            i += 1


# correct_skew(r"C:\Users\lenovo\PycharmProjects\SoftOCR_PFE\img")
