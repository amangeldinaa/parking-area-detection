# create a new path for images with reduced quality (50%) and simplified filenames
import os
import re
from PIL import Image

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def adjust_quality_and_save(input_path, output_path, quality=85):
    with Image.open(input_path) as img:
        img.save(output_path, 'JPEG', quality=quality)

def process_images(input_folder, output_folder, quality=85):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filenames = sorted(os.listdir(input_folder), key=natural_sort_key)
    for i, filename in enumerate(filenames):
        if filename.endswith(".jpg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"image{i + 1}.jpg")
            try:
                adjust_quality_and_save(input_path, output_path, quality)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

input_folder = 'prkarea-dataset/training'
output_folder = 'processed-data'
process_images(input_folder, output_folder, quality=50)
