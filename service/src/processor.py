import glob
import time
from copy import deepcopy

import imageio
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from batch_face import RetinaFace

from configs import *
from tools import *
from sort import *
from face_models import MobileFaceNet

class Processor:
    def __init__(self, device):
        if device == 'cuda':
            self.detector = RetinaFace(0)
        else:
            self.detector = RetinaFace(-1)
        self.tracker = Sort()
        self.model = get_model(MODEL_PATH, device)
        self.device = device
        self.faces_dict = {}
        self.classify_origin = False
        self.height = 380
        
    def detect_and_track(self, frames, oframes, scale):
        info = []
        faces = []
        classes = []
        orig_faces = []
        
        detected_frames = parse_out(self.detector(frames))
        
        assert len(detected_frames) == len(frames), (len(detected_frames), len(frames))
        
        if self.classify_origin:
            frames = oframes
        else:
            scale = 1
        
        for frame, out in zip(frames, detected_frames):
            detections = [face[0] for face in out]
            boxes = []
            if len(detections):
                tracked_objects = self.tracker.update(np.array(detections))
                for box in tracked_objects:
                    cls = box[-1]
                    box = [int(scale * b) for b in box[:-1]]
                    box = add_margin(*frame.shape[:2], box, EXTRA_MARGIN)
                    boxes.append((cls, box))
                    x1, y1, x2, y2 = box
                    faces.append(preprocess(frame[y1:y2, x1:x2]).unsqueeze(0))
                    classes.append(cls)
                    orig_faces.append(frame[y1:y2, x1:x2])
            info.append(boxes)
        return info, faces, classes, orig_faces

    def set_label(self, info, scores, frames, scale):
        processed = []
        ind = 0
        
        if self.classify_origin:
            scale = 1
            
        for boxes, frame in zip(info, frames):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for cls, box in boxes:
                score = scores[ind]
                ind += 1
                box = [int(scale * b) for b in box]
                x1, y1, x2, y2 = box
                copy_frame = deepcopy(frame)
                nbox = add_margin(*frame.shape[:2], box, 20)
                nx1, ny1, nx2, ny2 = nbox
                if cls in self.faces_dict:
                    self.faces_dict[cls].append((score, copy_frame[ny1:ny2, nx1:nx2]))
                else:
                    self.faces_dict[cls] = [(score, copy_frame[ny1:ny2, nx1:nx2])]
                label, color1, color2 = get_label(score)
                label = f"{label}"
                cv2.putText(frame, label, (x1, y1 + 30), cv2.FONT_HERSHEY_COMPLEX, 1.2, color1, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color2, 2)
            processed.append(frame)
        return processed
    
    def scale_frames(self, frames):
        scaled = []
        for frame in frames:
            scale = self.height / frame.shape[1]
            scaled.append(cv2.resize(frame, (0, 0), fx=scale, fy=scale))
        return scaled, 1 / scale
        
    def regress(self, faces):
        num_iter = len(faces) // R_BATCH_SIZE
        mod = len(faces) % R_BATCH_SIZE
        
        scores_all = []
        
        for i in range(num_iter):
            scores = self.model(faces[i * R_BATCH_SIZE: (i + 1) * R_BATCH_SIZE].to(self.device)).detach().cpu().numpy().reshape(-1)
            scores_all.extend(scores)
        if mod > 0:
            scores = self.model(faces[-mod:].to(self.device)).detach().cpu().numpy().reshape(-1)
            scores_all.extend(scores)
        
        assert len(faces) == len(scores_all), (len(faces), len(scores_all), num_iter, mod)
        return scores_all
        
    def __call__(self, frames):
        processed = []
        scaled_frames, scale = self.scale_frames(frames)
        info, faces, classes, orig_faces = self.detect_and_track(scaled_frames, frames, scale)
        if len(faces):
            faces = torch.cat(faces, dim=0)
            scores = self.regress(faces)
            processed = self.set_label(info, scores, frames, scale)
        return processed, (scores, classes, orig_faces)
