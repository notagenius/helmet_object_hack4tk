# helmet_object_hack4tk from Team ECC

IMAGE/VIDEO/LIVE Detector for TIM RUPP

#### 0. Utility:

code is minimized with openCV DNN read from darknet, and enough platform-freedom
pretained model provided

detection in image:
```
python3 Helmet_detection_YOLOV3.py -i ./test_input/test_working_helmet.jpg
```

detection in video:
```
python3 Helmet_detection_YOLOV3.py -v ./test_input/helmet.mp4
```

detection in webcam live:
```
python3 Helmet_detection_YOLOV3.py -c
```
there are flag in code you can set: 
for live demo effect
1) visual_result
2) save_result
3) resize_factor  


#### 1. Envirnment (virtualenv) depedencies or not

pip install -r requirements.txt

#### 2. Model problem

I found this model is not realiable during hackathon for it greatly mis-classficated human hair to Helmet.
I believe this can be removed by deeper training with more datasets.
made search on helmet open datasets, not very promissing,

the chinese one in this is the best, I wanted to do a good model for TIM RUPP for submission but unfornately,

till you are facing LIVE demo I am still not able to get any time for that. (I am very sorry)

https://github.com/wujixiu/helmet-detection (and it offeres other networks pretain model too)

But to address the performance during hackathon, YOLOv3 was choosen.

#### 3. if you wanna refine the model

training module: https://github.com/AlexeyAB/darknet

handy labelling tool: https://github.com/AlexeyAB/Yolo_mark
