import glob
import time
from copy import deepcopy

import imageio
import cv2
import PIL
import torch
import torch.nn as nn
from torchvision import transforms

from configs import *
from face_models import MobileFaceNet


resize = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

def preprocess(frame, transform=resize):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = PIL.Image.fromarray(img)
    sample = transform(im_pil)
    return sample

def get_model(path, device):
    feature_extractor = MobileFaceNet(512)
    model = nn.Sequential(
        feature_extractor,
        nn.Linear(512, 1)
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    return model

def get_label(score):
    score = 1 - score
    color2 = [50, 255, 255]
    if score < 0.48:
        color1 = [255, 0, 0]
        return "low", color2, color1
    elif score > 0.7:
        color1 = [0, 255, 0]
        return "high", color2, color1
    
    color1 = [255, 255, 0]
    return "medium", color2, color1

def parse_out(resp, score_thr=SCORE_THR, area_thr=AREA_THR):
    outs = []
    for faces in resp:
        f_outs = []
        for face in faces:
            box, landmarks, score = face
            box = [int(p) for p in box]
            left_eye, right_eye, nose = landmarks[:3]
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > area_thr and score > score_thr:
                f_outs.append((box, landmarks, score))
        outs.append(f_outs)
    return outs

def add_margin(h, w, box, margin=0):
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - margin)
    x2 = min(x2 + margin, w)
    y1 = max(0, y1 - margin)
    y2 = min(y2 + margin, h)
    return [x1, y1, x2, y2]
