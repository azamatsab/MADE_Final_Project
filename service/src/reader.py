import time

import cv2
import numpy as np

from configs import STEP, BATCH_SIZE


def read_video(path, processor, length=None):
    faces_dict = {}
    frames = []
    batch = []
    f_counter = 0
    
    show_count = 0
    
    cap = cv2.VideoCapture(path)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            f_counter += 1
            if f_counter % STEP == 0:
                batch.append(frame)
            if len(batch) == BATCH_SIZE:
                batch = processor(batch)
                frames += batch
                batch = []
        else:
            break
        if length == f_counter:
            break
    cap.release()
    return frames
