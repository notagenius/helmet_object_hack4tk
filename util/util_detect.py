import cv2
import sys
import numpy as np
import os.path

confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416

frame_count = 0 
frame_count_out=0

def drawPred(frame, classes, classId, conf, left, top, right, bottom):
    global frame_count
    cv2.rectangle(frame, (left, top), (right, bottom), (191, 255, 0), 3)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])


    label_name,label_conf = label.split(':')
    if label_name == 'Helmet':
        cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
        frame_count+=1
    if(frame_count> 0):
        return frame_count


def postprocess(frame, outs, classes, frame_count, fn):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global frame_count_out
    frame_count_out=0
    classIds = []
    confidences = []
    boxes = []
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    count_person=0
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        frame_count_out = drawPred(frame,classes,classIds[i], confidences[i], left, top, left + width, top + height)
        my_class='Helmet'
        unknown_class = classes[classId]

        if my_class == unknown_class:
            count_person += 1
    print("helmet detected:"+str(frame_count_out))
    return frame
