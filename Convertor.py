from pdf2image import convert_from_path
import os


def convert(file):
    pages = convert_from_path(f'{file}', 300, poppler_path=r"C:\Program Files\poppler-21.03.0\Library\bin")
    i = 0
    for page in pages:
        i += 1
        # Check if path exists or not
        if not os.path.exists(rf'C:\Users\lenovo\PycharmProjects\SoftOCR_PFE\Images'):
            os.mkdir(rf'C:\Users\lenovo\PycharmProjects\SoftOCR_PFE\Images')
        # Save each page as an image
        page.save(rf'C:\Users\lenovo\PycharmProjects\SoftOCR_PFE\Images\out{i}.png', 'PNG')
