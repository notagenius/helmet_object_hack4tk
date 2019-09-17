
### Image/Video/Live Helmet Detector for Tim Rupp, 

out-of-box demo ready solution. 

<img src="./result/test_larger.jpg" width="300"  />

### 0. USAGE:

deployed with openCV DNN ReadNetFromDarknet, and enough platform-freedom.
pretained model provided, quick but not well coded but functional.

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
python3 Helmet_Detector.py -c
```
flags in code you can set: 

| FLAGS  | DEFAULT |DESCRIPTION|
| ------------- | ------------- |------------- |
| visual_result  | True  |visualized the result with opencv|
| save_result  | True  | save the result as jpg or avi  |
| resize_factor   | 0.3 | resize video/webcam input resolution |



### 1. ENVIRNMENT INSTALLATION

developped with python3
```
pip3 install -r requirements.txt
```

### 2. MODEL PROBLEM

This model is our hackathon model, it works fine. but it has its own problem for it greatly mis-classficates human hair to Helmet.
this can be removed by deeper training with more datasets and taking a care on data augumentation.
made search on helmet open datasets the chinese one is the best, I wanted to do a good model for Tim for submission but unfornately I am still not able to get any time for that. (I am very sorry, to make a detect refinement on this model, the training at least need to run 40 hours + on GPU)

to address the performance speed on CPU, YOLOv3 was chosen.

### 3. DATASETS:

https://github.com/wujixiu/helmet-detection 

### 4. MODEL TRAINING

training module: https://github.com/AlexeyAB/darknet

a handy labelling tool: https://github.com/AlexeyAB/Yolo_mark
