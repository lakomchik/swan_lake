import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import PIL
from PIL import Image, __version__
PIL.PILLOW_VERSION = __version__
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage

class SwanDataset(Dataset):
    def __init__(self, root_dir, transforms):
        super().__init__()
        self.root_dir = root_dir
        self.folder_names = os.listdir(self.root_dir)
        self.images = []
        self.masks = []
        self.labels = []
        self.images_path = 'images'
        self.masks_path = 'masks'
        self.transforms = transforms
        for label, folder_name in enumerate(self.folder_names):
            folder_path = os.path.join(self.root_dir, folder_name)
            image_path = os.path.join(folder_path, self.images_path)
            mask_path = os.path.join(folder_path, self.masks_path)
            image_files = os.listdir(image_path)
            mask_files = os.listdir(mask_path)
            image_files.sort()
            mask_files.sort()
            for image_file in image_files:
                image_name = os.path.splitext(image_file)[0]
                mask_file = image_name + ".png"
                if mask_file in mask_files:
                    image = Image.open(os.path.join(image_path, image_file))
                    mask = Image.open(os.path.join(mask_path, mask_file))
                    if self.transforms is not None:
                        image = self.transforms(image)
                        mask = self.transforms(mask)
                    else:
                        image = ToTensor(image)
                        mask = ToTensor(mask)
                    self.images.append(image)
                    self.masks.append(mask)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        label = self.labels[idx]

        return image, mask, label
    
    def visualize_sample(self, idx):
        image, mask, label = self[idx]

        to_pil = ToPILImage()

        image = to_pil(image)
        mask = to_pil(mask)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')
        axes[1].imshow(mask)
        axes[1].set_title('Mask')
        axes[1].axis('off')
        plt.suptitle('Label: {}'.format(label))
        plt.show()

    def help(self):
        methods = [method_name for method_name in dir(self) if callable(getattr(self, method_name))]
        print("Available methods in SwanDataset:")
        for method_name in methods:
            print("- {}".format(method_name))


class SwanDataLoader():
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0):
        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)