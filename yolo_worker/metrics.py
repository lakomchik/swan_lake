from ultralytics import YOLO
import os
import cv2

model=YOLO('yolo_worker/weights.pt')
dataset= 'train/images/'

true_ans=0
all_ans=0

print(len(os.listdir(dataset)))

for img_path in sorted(os.listdir(dataset)):
    img=cv2.imread(dataset+img_path)
    result=model(img)
    obj=[0,0,0]
    for i in result[0].boxes:
        obj[int(i.cls)]+=1
    if obj==[0,0,0]:
        print('detected nothing')
    else:
        if int(img_path[0])==obj.index(max(obj)):
            true_ans+=1
            print('right answer for '+img_path)
        else:
            print('wrong answer for '+img_path)
    all_ans+=1
print('true ans:', true_ans)
print('all ans:', all_ans)
