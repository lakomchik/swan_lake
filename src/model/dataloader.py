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
import albumentations as A
from albumentations.pytorch import ToTensorV2


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class SwanDataset(Dataset):
    def __init__(self, root_dir, description_df, transforms=None):
        super().__init__()
        self.root_dir = root_dir
        self.description_df = description_df
        self.labels = self.description_df.swan_id.to_list()
        self.images = self.description_df.image_name.to_list()
        self.masks = self.description_df.mask_name.to_list()
        self.images_path = os.path.join(root_dir, 'images')
        self.masks_path = os.path.join(root_dir,'masks')
        self.transforms = transforms

    def __len__(self):
        return len(self.description_df)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.images_path, self.images[idx]))
        mask = Image.open(os.path.join(self.masks_path, self.masks[idx]))
        image = np.array(image)
        mask = np.array(mask)
        if self.transforms is not None:
            res = self.transforms(image=image,mask=mask)
            image = res['image']
            mask = res['mask']
        else:
            basic_transforms = A.Compose(
                [
                    A.Normalize(mean=(0.490, 0.450, 0.400),std=(0.230, 0.225, 0.225)),
                    ToTensorV2()
                ]
            )
            res = basic_transforms(image = image, mask = mask)
            image = res['image']
            mask = res['mask']
        label = self.labels[idx]

        return image, mask.type(torch.LongTensor), label
    
    def visualize_sample(self, idx):
        image, mask, label = self[idx]
        unnorm = UnNormalize(mean=(0.490, 0.450, 0.400),std=(0.230, 0.225, 0.225))
        image = unnorm(image)
        image = image.permute(1, 2, 0).numpy()
        mask = mask.numpy()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')
        axes[1].imshow(mask, cmap = 'gray')
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