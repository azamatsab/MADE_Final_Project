import sys
import time
import logging

import cv2
import numpy as np
from multiprocessing import Queue
from threading import Thread

from configs import STEP, BATCH_SIZE, QUEUE_SIZE


VIDEO_TYPE = "video"

def read_video(path, processor, length=None):
    frames = []
    batch = []
    f_counter = 0
    cap = cv2.VideoCapture(path)
    if (cap.isOpened()== False):
        logging.warning("Error opening video stream or file")
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


class Writer(Thread):
    def __init__(self, cap, reader):
        Thread.__init__(self)
        self.cap = cap
        self.reader = reader

    def run(self):
        f_counter = 0
        if (self.cap.isOpened() == False):
            logging.warning("Error opening video stream or file")
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                f_counter += 1
                if f_counter % STEP == 0:
                    self.reader.put(frame, VIDEO_TYPE)
            else:
                break
        self.cap.release()

class Reader(Thread):
    def __init__(self, processor):
        Thread.__init__(self)
        self.processor = processor
        self.inp_queue = Queue(QUEUE_SIZE)
        self.out_video = Queue(QUEUE_SIZE)
        self.out_stream = Queue(QUEUE_SIZE)
        self.batch = []
        self.sources = []

    def put(self, frame, packet_type):
        self.save_put((packet_type, frame), self.inp_queue)

    def save_put(self, data, queue):
        try:
            queue.put_nowait(data)
        except:
            queue.get()
            queue.put(data, True, 0.001)

    def get(self, queue):
        if not queue.empty():
            return queue.get()
        return None

    def get_video_frame(self):
        return self.get(self.out_video)

    def get_stream_frame(self):
        return self.get(self.out_stream)

    def process_batch(self):
        batch = self.processor(self.batch)
        
        for packet_type, frame in zip(self.sources, batch):
            if packet_type == VIDEO_TYPE:
                self.save_put(frame, self.out_video)
            else:
                self.save_put(frame, self.out_stream)
        
        self.batch = []
        self.sources = []

    def run(self):
        f_counter = 0
        while True:
            if self.inp_queue.empty():
                time.sleep(0.01)
                if len(self.batch):
                    self.process_batch()
            else:
                try:
                    for _ in range(STEP):
                        packet_type, frame = self.inp_queue.get()
                    self.batch.append(frame)
                    self.sources.append(packet_type)
                    if len(self.batch) == BATCH_SIZE:
                        self.process_batch()
                except Exception as e:
                    logging.warning(e)
