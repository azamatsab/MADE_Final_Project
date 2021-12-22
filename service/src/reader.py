import time
import logging

import cv2
import numpy as np
from multiprocessing import Queue
from threading import Thread

from configs import STEP, BATCH_SIZE, QUEUE_SIZE


def read_video(path, processor, length=None):
    frames = []
    batch = []
    f_counter = 0
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

class Reader(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.inp_queue = Queue(QUEUE_SIZE)
        self.out_video = Queue(QUEUE_SIZE)
        self.out_stream = Queue(QUEUE_SIZE)
        self.batch = []
        self.sources = []

    def process_batch(self):
        batch = processor(self.batch)
        self.batch = []
        self.sources = []
        
        for packet_type, frame in zip(self.sources, batch):
            if packet_type == "video":
                self.out_video.put(frame)
            else:
                self.out_stream.put(frame)

    def run(self):
        if self.inp_queue.empty():
            time.sleep(0.005)
            if len(self.batch):
                self.process_batch()
        else:
            try:
                packet_type, frame = self.packets.get()
                self.batch.append(frame)
                self.sources.append(packet_type)
                if len(self.batch) == BATCH_SIZE:
                    self.process_batch()
            except Exception as e:
                logging.warning(e)
