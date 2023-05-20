from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, path_to_weights = 'yolo_worker/weights_m.pt'):
        '''
        this generates instance of network, you can use your own path to weights
        
        '''
        self.model=YOLO(path_to_weights)

    def recognize(self, img, return_box=False):
        '''
        returns array w normalized class predictiton
        '''
        result=self.model()
        obj=[0,0,0]
        for i in result[0].boxes:
            obj[int(i.cls)]+=1
        obj=np.array(obj)
        obj=obj/np.sum(obj)
        return obj


