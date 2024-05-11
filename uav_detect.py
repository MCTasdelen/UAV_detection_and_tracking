import torch
import cv2
import cvzone
import math
import platform
import pathlib

plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')
print(device)
model = torch.hub.load('yolov5', 'custom', 'last.pt', source='local').to(device)

cap = cv2.VideoCapture("Test_UAV_video.mp4")
cap.set(3, 1280)
cap.set(4, 720)



while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_tensor = torch.from_numpy(img).to(device)
    results = model(img)


    boxes = results.xyxy[0].cpu().numpy()
    labels = results.names


    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = f'{labels[int(cls)]} {conf:.2f}'
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        img = cv2.putText(img, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 55, 105), 1)
        obj_centre_x = int((x1+x2)/2)
        obj_centre_y = int((y1+y2)/2)

        img = cv2.rectangle(img, (100,64),(700,380),(0,0,255),3)

        img = cv2.line(img,(400,222),(obj_centre_x,obj_centre_y),(255,0,0),1)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



    cv2.imshow("Image", img)
    cv2.waitKey(1)


cv2.imshow("Image", img)




cap.release()
cv2.destroyAllWindows()
