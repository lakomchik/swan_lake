from yolo_worker.inference import YOLODetector

from src.model.mobile_inf_class import MViT

from src.model.yolo_vit_inf_class import YoloVitPredictor
import cv2
import numpy as np
from PIL import Image
import os


# OURS 0: KLIKUN , 1: MALIY, 2: shipun
# ORGS 1: MALIY, KLIKUN: 2, SHIPUN: 3

label_mapper = {


}


class Ansamble():
    def __init__(self) -> None:
        self.clfs = [YOLODetector(),
                     MViT(),
                     YoloVitPredictor()]

    def forward(self, image):
        preds = 0
        probas = np.zeros(3, dtype=float)
        for clasifier in self.clfs:
            _, prob = clasifier.forward(image)
            probas += prob
        probas = probas/np.sum(probas)
        return np.argmax(probas), probas

    def multiple_inference(self, list_of_paths):
        out_list = [0] * len(list_of_paths)

        for i, path in enumerate(list_of_paths):
            image = Image.open(path)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image = np.array(image)
            _, out = self.forward(image)
            out_list[i] = out
        return list(zip(list_of_paths, out_list))


if __name__ == "__main__":
    # img = cv2.imread("data/klikun/images/24_2spring_Mizinenko_6674.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    # res = model.forward(img)
    model = Ansamble()
    result = model.multiple_inference([os.path.join('data/small_dataset/Merged/images', filename)
                                       for filename in os.listdir('data/small_dataset/Merged/images')[:10]])
    print(result)
