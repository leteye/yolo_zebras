# yolo_zebras

**Zebras aerial video tracking with YOLOv8**

The Project developed for tracking zebras in a wild on video recorded by UAV. The task is to estimate group spatial dynamics (changes in distances between individuals) during а whole video. Distances estimated for all individuals by "each VS each" scheme n = (n*(n-1))/2, where n is number of individuals succesfully tracked on the frame. All values were normalized with body length of individual specified. The data were calculated for each frame and each succesfully tracked individual. Results consists of:
1. Two .csv tables (version for automated and for manual analysis), table columns are:
- frame (frame number);
- IDs (individuals IDs that succesfully tracked on the frame);
- Xc (x-coordinate for the approx body center of certain individual);
- Yc (same for Y-coordinate);
- Dists (normalized distances values)
2. Annotated video with results (IDs numbers and approximated center points)
3. Same data as video as directory with all frames in .jpg
  
**Sample results could be accessed by request.**

![results.png](/blob/results.png)
_0, 60 and 120 frame of output video_

YOLO model training and inference as tracker inference was conducted on downsampled and cropped original video (from 2700x1520 to 960x760). Module utils.py include several helper functions for data preprocessing & annotation stages.

## Task "reverse-decomposition":
- Record distances in .csv <-- calculate distances
- Calculate distances <-- find approx centers
- Find approx centers <-- find boxes (or contours) of objects
- Find boxes (or contours) of objects <-- setup tracker & inference
- Setup tracker & inference <-- train YOLO model (and check metrics)
- Train & validate YOLO model <-- annotate data
- Annotate data <-- preprocess & select data from original video

### Tracker setup
Bytetrack tracker type were selected, parameters were adapted to current YOLO model performance, specific data and task. All parameters saved in "bytetrack-2.yaml" file. Original tracker .yaml files could be accessed from https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers

### The YOLO model
Tracker is based on pretrained middle-size YOLOv8 detection model (yolov8m.pt, weights could be acessed on https://github.com/ultralytics/ultralytics#models Ultralytics page). Model were trained on 382 760x760 RGB tiles, and validated on 47 such tiles to single-class detection task with batch size of 20 during 60 epochs.

**Model weights could be accessed by reques**t.

### Data annotation
Train & val data were annotated witn VIA, v.2 tool: https://www.robots.ox.ac.uk/~vgg/software/via/ Labels were generated with "VIA2YOLO_detect" helper function. Train and val sets were selected and splitted manually for the sake of maximum data diversity and relevance. Helper "labels4images" function split labels accordingly.

### Data preprocessing
Tiles were generated with "click_sampler" helper function - cropped from selected frames of downsampled original video ("video2frames" and "frames2video" helper functions).

### Requirements
Scripts were tested on machine with Nvidia GeForce RTX 3060 12Gb, AMD Ryzen 5700X and 64Gb RAM (Ubuntu 20.04 LTS, python 3.9). All requirements could be installed with terminal: `pip install -r requirements.txt`

## Contributions
All the work performed by **merr**. Original data were obtained from Andrey Gilyov.

## Aknowledgements
Work conducted in collaboration with Andrey Gilyov Research Group form Dept of Vertebrate Zoology, Saint-Petersburg State University in 2023.
