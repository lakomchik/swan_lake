import torch
from torch import nn
import numpy as np
import timm
from PIL import Image
from torchvision import transforms
import os
from tqdm.autonotebook import tqdm
from albumentations import Resize, Compose, ToFloat
from albumentations.pytorch import ToTensorV2
import sys
import pathlib
import pandas as pd
chkp_path = os.path.join(os.getcwd(), 'src', 'checkpoints')


class MViT():
    def __init__(self, weights_path=os.path.join(chkp_path, 'best_mobile.pt')) -> None:

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'

        self.model = timm.create_model(
            'mobilevitv2_075.cvnets_in1k', pretrained=True, num_classes=3)

        num_in_feat = self.model.get_classifier().in_features

        self.model.fc = nn.Sequential(nn.BatchNorm1d(num_in_feat),
                                      nn.Linear(
                                          in_features=num_in_feat, out_features=512),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),
                                      nn.Dropout(0.4),
                                      nn.Linear(in_features=512, out_features=3))

        dct = torch.load(weights_path)

        self.model.load_state_dict(dct)
        self.model.to(self.device)
        self.model.eval()
        self.transforms = Compose([
            Resize(256, 256),
            ToFloat(),
            ToTensorV2()
        ])

    def forward(self, image):
        with torch.no_grad():
            image = np.array(image)
            image_transformed = self.transforms(
                image=image)['image'].unsqueeze(0)
            outputs = self.model(image_transformed).softmax(dim=1)
            out = outputs.argmax(1)[0]
            return out.detach().cpu().numpy(), outputs.cpu().numpy().reshape(-1)

    def multiple_inference(self, list_of_paths):
        out_list = [0] * len(list_of_paths)
        with torch.no_grad():
            self.model.eval()
            for i, path in tqdm(enumerate(list_of_paths)):
                image = np.array(Image.open(path))
                image_transformed = self.transforms(
                    image=image)['image'].unsqueeze(0).to(self.device)
                out = self.model(image_transformed).softmax(
                    1).detach().cpu().numpy()
                out_list[i] = out
        return list(zip(list_of_paths, out_list))


def main():
    mf = model_inference()
    result = mf.multiple_inference([os.path.join('./train_dataset_Минприроды/shipun/images/', filename)
                                   for filename in os.listdir('./train_dataset_Минприроды/shipun/images/')])
    print(result)


if __name__ == '__main__':
    main()
