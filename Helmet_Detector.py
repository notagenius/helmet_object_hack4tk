#####################################################################
## image demo: python3 Helmet_Detector.py -i ./test_input/test_working_helmet.jpg
## video demo: python3 Helmet_Detector.py -v ./test_input/helmet.mp4
## webcam live demo: python3 Helmet_Detection.py -c
#####################################################################
from time import sleep
import cv2
import sys
import numpy as np
import os.path
from util import util_detect as util_detect

#### flag you can set: visualize result by imshow, and save by imsave
visual_result = True
save_result = True

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

frame_count = 0 
frame_count_out=0

confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416

classesFile = "util/obj.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = "util/yolov3_helmet.cfg";
modelWeights = "util/yolov3_helmet.weights";

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

if len(sys.argv) > 1:
    if sys.argv[1] == "-i":
      frame = cv2.imread(sys.argv[2])
      frame_count =0
      blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
      net.setInput(blob)
      outs = net.forward(getOutputsNames(net))
      result = util_detect.postprocess(frame, outs, classes, frame_count,sys.argv[1])
      if save_result == True:
        path = './result/'
        frame_name=os.path.basename(sys.argv[2])
        cv2.imwrite(str(path)+frame_name, result)
      if visual_result == True:
        cv2.imshow('image',frame)
        cv2.waitKey()
    
    if sys.argv[1] == "-v":
      # resize factor you can set won't speed up detection, but for visualization
      video_factor = 0.3
      cap=cv2.VideoCapture(sys.argv[2])
      ret, frame = cap.read()
      frame = cv2.resize(frame, None, fx = video_factor, fy = video_factor)
      height, width, depth = frame.shape
      id_count = 0
      if save_result == True:
        path = './result/'
        frame_name=os.path.basename(sys.argv[2])
        print(str(path)+frame_name)
        out = cv2.VideoWriter(str(path)+frame_name, cv2.VideoWriter_fourcc('M','J','P','G'), 25, (width,height))
      while(True):
          person=[]
          ret, frame = cap.read()
          frame = cv2.resize(frame, None, fx = video_factor, fy = video_factor)
          frame_count = 0
          id_count = id_count + 1
          blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
          net.setInput(blob)
          print("video_frame_count:" + str(id_count))
          outs = net.forward(getOutputsNames(net))
          result = util_detect.postprocess(frame, outs, classes, frame_count,sys.argv[1])
          t, _ = net.getPerfProfile()
          if save_result == True:
            out.write(result)
          if visual_result == True:
            cv2.imshow('video',result)
            cv2.waitKey(1)
      cap.release()
      out.release()

    if sys.argv[1] == "-c":
      # resize factor you can set won't speed up detection, but for visualization
      webcam_factor = 0.3
      cap = cv2.VideoCapture(0)
      ret, frame = cap.read()
      frame = cv2.resize(frame, None, fx = webcam_factor, fy = webcam_factor)
      height, width, depth = frame.shape
      id_count = 0
      while(True):
          person=[]
          ret, frame = cap.read()
          frame = cv2.resize(frame, None, fx = webcam_factor, fy=webcam_factor)
          frame_count = 0
          id_count = id_count + 1
          blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
          net.setInput(blob)
          print("video_frame_count:" + str(id_count))
          outs = net.forward(getOutputsNames(net))
          result = util_detect.postprocess(frame, outs, classes, frame_count,sys.argv[1])
          t, _ = net.getPerfProfile()
          if visual_result == True:
            cv2.imshow('webcam',result)
            cv2.waitKey(1)
      cap.release()
      out.release()

    
else:
    print("please send input.")
    print("to test images: -i path_to_image")
    print("to test video: -v path_to_video")
    print("to test camera: -c")


