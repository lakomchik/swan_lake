import torch
from torch import nn 
import numpy as np
import timm
from PIL import Image
from torchvision import transforms
from torchvision.models import regnet_y_3_2gf, RegNet_Y_3_2GF_Weights
import os
from tqdm.autonotebook import tqdm
from albumentations import Resize, Compose, ToFloat
from albumentations.pytorch import ToTensorV2
import sys
import pathlib
import pandas as pd
chkp_path = os.path.join(os.getcwd(),'src','checkpoints')

class regnet_inference():
    def __init__(self, weights_path = os.path.join(chkp_path,'best_regnet.pt')) -> None:
        
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'

        self.model = regnet_y_3_2gf(weights = RegNet_Y_3_2GF_Weights)
        
        dct = torch.load(weights_path)

        self.model.load_state_dict(dct)
        self.model.to(self.device)

        self.transforms =  Compose([
                    Resize(256,256),
                    ToFloat(),
                    ToTensorV2()
                ])

    # def single_inference(self,image):
    #     image = np.array(image)
    #     image_transformed = self.transforms(image=image)['image'].unsqueeze(0)
    #     out = self.model(image_transformed).argmax(1)[0]
    #     return out.detach().cpu().numpy()
    
    def multiple_inference(self,list_of_paths):
        out_list = [0] * len(list_of_paths) 
        with torch.no_grad():
            self.model.eval()
            for i,path in tqdm(enumerate(list_of_paths)):
                image = np.array(Image.open(path))
                image_transformed = self.transforms(image=image)['image'].unsqueeze(0).to(self.device)
                out = self.model(image_transformed).softmax(1).detach().cpu().numpy()
                out_list[i] = out
        return list(zip(list_of_paths,out_list))

# def main():
#     regnet = regnet_inference()
#     result = regnet.multiple_inference([os.path.join('./Dataset/maliy/images/',filename) for filename in os.listdir('./Dataset/maliy/images/')])
#     print(result)

# if __name__ == '__main__':
#     main()