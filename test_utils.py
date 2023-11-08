#---TEST UTILS FUNCTIONS---

import cv2
import os
import natsort as ns
import utils as ut  # import own utils module

# get_video_specs
path = '15s.mp4'
specs = ut.get_video_specs(path)
print(specs)

# video2frames
path = '15s.mp4'
out = 'z2_15s_960x760'
cap = cv2.VideoCapture(path)
new_x, new_y = int(cap.get(3)), int(cap.get(4))
frames_processed = ut.video2frames(path, out, 1, new_x, new_y, 0, 280, 760, 1240)
print(frames_processed)

# frames2video
ut.frames2video('z2_15s_960x760', '15s_all.mp4', 24)

# click_sampler for single pic
ut.click_sampler('z1_frames_960x760/z1-0.jpg', 760, 'tiles_click')

# click_sampler for series of pic in the folder
src = 'click_sampler_test'
for file in ns.natsorted(os.listdir(os.getcwd() + '/' + src)):
    ut.click_sampler(src + '/' + file, 640, 'frames_all_tiles')

# VIA2YOLO_detect
data = 'z2_382_47_m_80EP_071023.csv'
codes = {'zebra': 0}
out_dir = 'out'
ut.VIA2YOLO_detect(data, codes, out_dir, (760, 760))

# labels4images
root = 'train_val'
ut.labels4images(root, 'images', 'labels')
