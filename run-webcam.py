from ultralytics import YOLO
import cv2

model=YOLO('best.pt')

src='http://192.168.1.3:8080/video'
capture=cv2.VideoCapture(src)

while True:
    success, frame=capture.read()
    if success:
        result=model(frame, stream=True)
        for i in result:
            annoted_frame=i.plot()
            cv2.imshow('mudinchu',annoted_frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    else:
        break





