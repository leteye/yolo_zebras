# Test utils

import torch
from ultralytics import YOLO
import lap  # dependency for YOLO tracker
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import os
import shutil
import warnings
import random
import matplotlib.cbook as cbook
import natsort as ns
import json
import itertools as it

import utils as ut  # import own utils module

# to reproduce same results fixing the seed and hash
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# PT version, CPU/GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'torch version: {torch.__version__}, device: {device}')

#---TEST UTILS FUNCTIONS---

# get_video_specs
path = '../../datasets/zebras/Zebra2.mp4'
specs = ut.get_video_specs(path)
print(specs)

# video2frames
path = '../../datasets/zebras/Zebra1.mp4'
out = '/z1_frames_960x760/'
cap = cv2.VideoCapture(path)
new_x, new_y = int(cap.get(3) / 2), int(cap.get(4) / 2)
frames_processed = ut.video2frames(path, out, 1, new_x, new_y, 0, 280, 760, 1240)
print(frames_processed)

# frames2video
ut.frames2video('/frames_960x760/z1/all/', 'all.mp4', 24)

# click_sampler for single pic
ut.click_sampler('z1_frames_960x760/z1-0.jpg', 760, 'tiles_click')

# click_sampler for series of pic in the folder
src = 'z1_frames_960x760'
for file in ns.natsorted(os.listdir(os.getcwd() + '/' + src)):
    ut.click_sampler(src + '/' + file, 640, 'frames_all_tiles')

# VIA2YOLO_detect
data = './detect/annots/z1/z1_125_16_081023.csv'
codes = {'zebra': 0}
out_dir = 'out/'
ut.VIA2YOLO_detect(data, codes, out_dir, (760, 760))

# labels4images
root = 'detect/z11/'
ut.labels4images(root, 'images', 'labels')
