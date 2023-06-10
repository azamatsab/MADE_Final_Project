import sys
import time
import logging
import random
from collections import defaultdict

import cv2
import numpy as np
from multiprocessing import Queue
from threading import Thread
import imageio

from configs import STEP, BATCH_SIZE, QUEUE_SIZE
from tools import id_generator


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
        self.fps = cap.get(cv2.CAP_PROP_FPS)

    def run(self):
        f_counter = 0
        if (self.cap.isOpened() == False):
            logging.warning("Error opening video stream or file")
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                f_counter += 1
                if f_counter % STEP == 0:
                    self.reader.put(frame, VIDEO_TYPE, 1000 * self.fps / f_counter)
            else:
                break
        self.cap.release()

class Info:
    def __init__(self):
        self.classes = np.array([])
        self.scores = np.array([])
        self.timestamps = np.array([])
        self.faces = defaultdict(list)
        self.gif_length = 20

    def add(self, scores, classes, faces, timestamps):
        self.classes = np.concatenate([self.classes, classes])
        self.scores = np.concatenate([self.scores, scores])
        self.timestamps = np.concatenate([self.timestamps, timestamps])

        for face, cls in zip(faces, classes):
            self.faces[cls].append(face)
            if len(self.faces[cls]) > self.gif_length:
                random.shuffle(self.faces[cls])
                self.faces[cls].pop()

    def get_data(self):
        classes = set(list(self.classes))
        filenames = []
        for cls in list(classes):
            scores = self.scores[self.classes == cls]
            timestamps = self.timestamps[self.classes == cls]
            f_name = str(cls) + ".csv"
            with open(f_name, "w") as fout:
                for score, ts in zip(scores, timestamps):
                    fout.write("{},{}\n".format(score, ts))
            filenames.append(f_name)
        return filenames

    def get_gif(self):
        classes = list(set(list(self.classes)))
        filenames = []
        for cls in classes:
            faces = self.faces[cls]
            gif_path = id_generator() + ".gif"
            imageio.mimsave(gif_path, [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces], duration=0.5)
            filenames.append(gif_path)
        return filenames

class Reader(Thread):
    def __init__(self, processor):
        Thread.__init__(self)
        self.processor = processor
        self.inp_queue = Queue(QUEUE_SIZE)
        self.out_video = Queue(QUEUE_SIZE)
        self.out_stream = Queue(QUEUE_SIZE)
        self.batch = []
        self.sources = []
        self.timestamps = []

        self.video_info = Info()
        self.stream_info = Info()

    def put(self, frame, packet_type, timestamp):
        self.save_put((packet_type, frame, timestamp), self.inp_queue)

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
        batch, (scores, classes, faces, timestamps) = self.processor(self.batch, self.timestamps)
        
        for packet_type, frame in zip(self.sources, batch):
            if packet_type == VIDEO_TYPE:
                self.save_put(frame, self.out_video)
                self.video_info.add(scores, classes, faces, timestamps)
            else:
                self.save_put(frame, self.out_stream)
                self.stream_info.add(scores, classes, faces, timestamps)
            
            print(self.out_video.qsize(), self.inp_queue.qsize())
            sys.stdout.flush()

        self.batch = []
        self.sources = []
        self.timestamps = []

    def run(self):
        f_counter = 0
        while True:
            if self.inp_queue.empty():
                time.sleep(0.001)
                if len(self.batch):
                    self.process_batch()
            else:
                try:
                    for _ in range(STEP):
                        packet_type, frame, timestamp = self.inp_queue.get()
                    self.batch.append(frame)
                    self.sources.append(packet_type)
                    self.timestamps.append(timestamp)
                    if len(self.batch) == BATCH_SIZE:
                        self.process_batch()
                except Exception as e:
                    logging.warning(e)
