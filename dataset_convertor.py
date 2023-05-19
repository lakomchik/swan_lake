import os
import numpy as np
import cv2

# input_dir = "shipun/masks/"
# output_dir = "shipun/masks_new/"
# int_num=0


# input_dir = "klikun/masks/"
# output_dir = "klikun/masks_new/"
# int_num = 2

input_dir = "small/masks/"
output_dir = "small/masks_new/"
int_num = 1


for filename in os.listdir(input_dir):
    img_array = cv2.imread(os.path.join(input_dir, filename))

    colors = np.unique(img_array)
    for color in colors:

        if color == 0:
            continue
        mask = np.all(img_array == color, axis=-1)
        indices = np.where(mask)
        x_min, y_min = np.min(indices, axis=1)
        x_max, y_max = np.max(indices, axis=1)

        # Calculate middle, width, and height
        x_mid = x_min
        y_mid = y_min
        width = x_max - x_min
        height = y_max - y_min

        # Normalize coordinates
        x_norm = x_mid / img_array.shape[0]
        y_norm = y_mid / img_array.shape[1]
        width_norm = width / img_array.shape[0]
        height_norm = height / img_array.shape[1]

        if x_norm > 1 or y_norm > 1:
            print("overlimit!")
        # Write to output file
        with open(output_dir+filename[:-4]+'.txt', "a") as f:
            f.write(str(int_num)" {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                x_norm, y_norm, width_norm, height_norm))
