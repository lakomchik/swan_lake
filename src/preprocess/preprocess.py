import warnings
import cv2
import pandas as pd
from tqdm.autonotebook import tqdm
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import os
import random
import torch
import PIL
from PIL import Image, __version__
PIL.PILLOW_VERSION = __version__
warnings.filterwarnings('ignore')


def basic_preprocess(root_dir):
    # images_path = os.path.join(root_dir, '/Merged/images')
    # masks_path = os.path.join(root_dir, '/Merged/masks')

    total_swans_num = 0
    description_df = pd.DataFrame()

    images_path = root_dir + '/Merged/images/'
    masks_path = root_dir + '/Merged/masks/'
    print(root_dir)
    # folder_names = os.listdir('data')
    # folder_names.sort()
    folder_names = ['klikun', 'maliy', 'shipun']
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)
    for label, folder_name in enumerate(folder_names):
        folder_path = os.path.join(root_dir, folder_name)
        # folder_path = os.path.join(root_dir, folder_name)

        image_path = os.path.join(folder_path, 'images')
        mask_path = os.path.join(folder_path, 'masks')
        image_files = os.listdir(image_path)
        mask_files = os.listdir(mask_path)
        image_files.sort()
        mask_files.sort()
        
        for image_file, mask_file in tqdm(zip(image_files, mask_files)):
            image = Image.open(os.path.join(image_path, image_file))
            mask = Image.open(os.path.join(mask_path, mask_file))
            # skipping mismatching data
            if (image.size[0] != mask.size[0] and image.size[1] != mask.size[1]):
                continue
            if image.mode == "RGBA":
                image = image.convert("RGB")
            try:
                image.save(os.path.join(
                    images_path, str(total_swans_num)+'.jpg'))
                mask.save(os.path.join(
                    masks_path, str(total_swans_num)+'.png'))
                total_swans_num += 1
                new_row = {"swan_id": label, "image_name": str(
                    total_swans_num) + ".jpg", "mask_name": str(total_swans_num) + ".png"}
                description_df = description_df.append(
                    new_row, ignore_index=True)
            except:
                print(f"Error with image {image_file} and mask {mask_file}")
                break
    return description_df