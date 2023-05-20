from ultralytics import YOLO
import cv2
import numpy
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import torchvision.models as models
import timm


class YoloVitPredictor():
    def __init__(self, yolo_path, classifier_path, device='cpu') -> None:
        self.yolo = YOLO(yolo_path)
        self.classifier = timm.create_model(
            'mobilevitv2_075.cvnets_in1k', pretrained=True, num_classes=3)
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.eval()
        self.size = (256, 256)
        self.clf_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.device = device
        self.classifier = self.classifier.to(self.device)

    def single_inference(self, img):
        # yolo inference
        if (self.device == 'cpu'):
            yolo_output = self.yolo.predict(
                img, imgsz=640, device=self.device, verbose=False)
        else:
            yolo_output = self.yolo.predict(
                img, imgsz=640, verbose=False)
        boxes = np.array(yolo_output[0].boxes.xyxy.cpu().numpy(), dtype=int)
        clf_input = []
        for el in boxes:
            clf_input.append(self.clf_transforms(
                img[el[1]:el[3], el[0]:el[2], :]))
        batch = torch.stack(clf_input)
        with torch.no_grad():
            outputs = self.classifier(batch)
            predictions = torch.argmax(outputs, dim=1).to(self.device)
            result = torch.argmax(torch.bincount(predictions)).cpu().item()
            return result


if __name__ == "__main__":
    yolo_path = 'src/checkpoints/yolo_swan_detector.pt'
    mvit_path = 'src/checkpoints/MViT_single_swan.pth'
    model = YoloVitPredictor(yolo_path, mvit_path)
    img = cv2.imread("pics/4.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = model.single_inference(img)
    print(res)
