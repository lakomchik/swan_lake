import os
import random
import shutil

# Set the paths to the image and label folders
image_folder = 'train/images/'
label_folder = 'train/labels/'

# Create the test and validation folders
test_image_folder = 'test/images/'
test_label_folder = 'test/labels/'
val_image_folder = 'val/images/'
val_label_folder = 'val/labels/'

os.makedirs(test_image_folder, exist_ok=True)
os.makedirs(test_label_folder, exist_ok=True)
os.makedirs(val_image_folder, exist_ok=True)
os.makedirs(val_label_folder, exist_ok=True)

# Get a list of all the file names in the image folder
file_names = os.listdir(image_folder)

# Shuffle the file names randomly
random.shuffle(file_names)

# Calculate the number of files to move to the test and validation sets
test_size = int(0.1 * len(file_names))
val_size = int(0.1 * len(file_names))

# Move the first test_size files to the test set
for i in range(test_size):
    src_image_path = os.path.join(image_folder, file_names[i])
    src_label_path = os.path.join(label_folder, os.path.splitext(file_names[i])[0] + '.txt')
    dst_image_path = os.path.join(test_image_folder, file_names[i])
    dst_label_path = os.path.join(test_label_folder, os.path.splitext(file_names[i])[0] + '.txt')
    try:
        shutil.move(src_image_path, dst_image_path)
        shutil.move(src_label_path, dst_label_path)
    except:
        pass

# Move the next val_size files to the validation set
for i in range(test_size, test_size + val_size):
    src_image_path = os.path.join(image_folder, file_names[i])
    src_label_path = os.path.join(label_folder, os.path.splitext(file_names[i])[0] + '.txt')
    dst_image_path = os.path.join(val_image_folder, file_names[i])
    dst_label_path = os.path.join(val_label_folder, os.path.splitext(file_names[i])[0] + '.txt')
    try:
        shutil.move(src_image_path, dst_image_path)
        shutil.move(src_label_path, dst_label_path)
    except:
        pass