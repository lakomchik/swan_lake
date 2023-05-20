import os
import cv2
import numpy as np
import shutil

# Define the path to the folder containing the folders of images and masks
data_path = "source"

# Define the path to the folder where the labeled images will be saved
labels_path = "train/labels"

# Define the path to the folder where the original images will be copied
images_path = "train/images/"

# Loop through each folder in the data folder
class_num=0
for folder_name in sorted(os.listdir(data_path)):
    folder_path = os.path.join(data_path, folder_name)

    img_folder_path = os.path.join(folder_path, "images")
    mask_folder_path = os.path.join(folder_path, "masks")
    
    # Skip any non-folder items within the folder

    
    # Load the image and mask files
    image_files = [f for f in os.listdir(img_folder_path)]
    mask_files = [f for f in os.listdir(mask_folder_path)]
    
    # Iterate over each mask file
    num=0
    for mask_file_name in sorted(mask_files):
        mask_file_path = os.path.join(mask_folder_path, mask_file_name)
        label_file=open(os.path.join(labels_path,str(class_num)+"_"+str(num)+'.txt'),'w')
        print(mask_file_path)
        # Load the mask image and find unique colors
        mask_image = cv2.imread(mask_file_path)
        height,width,_ = mask_image.shape
        unique_colors = np.unique(mask_image)
        # Iterate over each unique color
        for color in unique_colors:
            if color==0:
                continue
            # Find the minimum and maximum coordinates for this color
            color_mask = np.all(mask_image == color, axis=-1)
            ys, xs = np.nonzero(color_mask)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            # Write the coordinates to the label file
            label_line = f"{folder_name} {x_min/width} {y_min/height} {(x_max - x_min)/width} {(y_max - y_min)/height}\n"
            label_file.write(label_line)
        label_file.close()
        
        # Copy the original image to the images folder, using sequence number as name
        shutil.copy(img_folder_path+'/'+mask_file_name[:-4]+'.jpg',images_path+str(class_num)+"_"+str(num)+'.jpg' )
        num+=1
    class_num+=1