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
chkp_path = os.path.join(os.getcwd(),'src','checkpoints')


class model_inference():
    def __init__(self, weights_path = os.path.join(chkp_path,'best_mobile.pt')) -> None:

        self.model = timm.create_model('mobilevitv2_075.cvnets_in1k', pretrained=True,num_classes=3)

        num_in_feat = self.model.get_classifier().in_features

        self.model.fc = nn.Sequential(nn.BatchNorm1d(num_in_feat),
                         nn.Linear(in_features=num_in_feat,out_features=512),
                         nn.ReLU(),
                         nn.BatchNorm1d(512),
                         nn.Dropout(0.4),
                         nn.Linear(in_features=512,out_features=3))
        
        dct = torch.load(weights_path)

        self.model.load_state_dict(dct)

        self.transforms =  Compose([
                    Resize(256,256),
                    ToFloat(),
                    ToTensorV2()
                ])

    def single_inference(self,image):
        image = np.array(image)
        image_transformed = self.transforms(image=image)['image'].unsqueeze(0)
        out = self.model(image_transformed).argmax(1)[0]
        return out.detach().cpu().numpy()


def main():
    mf = model_inference()
    result = mf.single_inference(Image.open('./train_dataset_Минприроды/Merged/images/0.jpg'))
    print(result)

if __name__ == '__main__':
    main()
