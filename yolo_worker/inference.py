import cv2
from ultralytics import YOLO
import numpy as np


class YOLODetector:
    def __init__(self, path_to_weights='yolo_worker/weights_m.pt'):
        '''
        this generates instance of network, you can use your own path to weights

        '''
        self.model = YOLO(path_to_weights)

    def forward(self, img, return_box=False):
        '''
        returns array w normalized class predictiton
        '''
        result = self.model.predict(img, verbose=False)
        obj = [0, 0, 0]
        for i in result[0].boxes:
            obj[int(i.cls)] += i.conf.cpu().item()
        obj = np.array(obj)
        obj = obj/np.sum(obj)
        return obj.argmax(), np.array(obj, dtype=float)


if __name__ == "__main__":
    img = cv2.imread("data/klikun/images/24_2spring_Mizinenko_6674.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model = YOLODetector()
    print(model.recognize(img))
