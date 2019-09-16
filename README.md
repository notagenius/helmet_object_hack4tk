### from Team ECC #hack4tk

Image/Video/Live Helmet Detector for Tim Rupp, 

out-of-box demo ready solution. 

<img src="./result/test_larger.jpg" width="300"  />

#### 0. Utility:

model is deployed with openCV DNN ReadNetFromDarknet, and enough platform-freedom.
pretained model provided, quick coded, dumb parameter setting and exceptions are not handled. 
but just try to make it functional.

detection in image:
```
python3 Helmet_Detector.py -i ./test_input/test_working_helmet.jpg
```

detection in video:
```
python3 Helmet_Detector.py -v ./test_input/helmet.mp4
```

detection in webcam live:
```
python3 Helmet_Detection.py -c
```
there are flag in code you can set: 

| flags  | default |
| ------------- | ------------- |
| visual_result  | True  |
| save_result  | True  |
| resize_factor   | 0.3 |



#### 1. Envirnment (virtualenv) depedencies

developped with python3

pip3 install -r requirements.txt

#### 2. Model problem

This model is our hackathon model, it works fine. but it has its own problem for it greatly mis-classficates human hair to Helmet.
this can be removed by deeper training with more datasets and taking a care on data augumentation.
made search on helmet open datasets the chinese one is the best, I wanted to do a good model for Tim for submission but unfornately I am still not able to get any time for that. (I am very sorry, to make a detect refinement on this model, the training at least need to run 40 hours + on GPU)

datasets:

https://github.com/wujixiu/helmet-detection (this one offeres other networks pretain model too)

to address the performance speed during hackathon on CPU, YOLOv3 was chosen.

#### 3. if you wanna refine the model with your own data

training module: https://github.com/AlexeyAB/darknet

a handy labelling tool: https://github.com/AlexeyAB/Yolo_mark

#### silly comments,

webcam code is i blinded coded for i have no webcam now, not likely it will have bug, but maybe
