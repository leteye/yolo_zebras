# YOLO train and inference script

import torch
from ultralytics import YOLO
import lap  # dependency for YOLO tracker
import numpy as np
import pandas as pd
import cv2
import os
import shutil
import random
import natsort as ns
import json
import itertools as it

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

# x-large model with default settings too big for 12GB video GPU (@batch=16, @RGB760px)
# detection model of middle size used (all models are in "models" dir)


def train(model_path, yml_data):
    """
    :param model_path: path to model.pt file
    :param yml_data: yaml config file for yolo training
    :return: validation metrics
    """
    model = YOLO(model_path)  # pretrained model
    model.train(data=yml_data, batch=16, epochs=20, imgsz=768)
    metrics = model.val()  # validate the model

    return metrics


def test(model_path, test_folder):
    """
    :param model_path: path to model.pt weights file

    :param folder: src folder for testing
    :return: test results
    """

    best_model = YOLO(model_path)
    results = best_model.predict(
        source=test_folder, save=True, save_txt=True, imgsz=768,
        line_thickness=1, save_conf=True,  # save_crop=True,
        iou=0.3, conf=0.5, exist_ok=False
    )
    return results


def yolo_track(tracker_cfg, model, src):
    """
    :param tracker_cfg: config yaml file for tracker
    :param model: path to model.pt weights file
    :param src: src video file
    :return: results
    """

    model = YOLO(model)
    model.to(device)
    results = model.track(
        source=src, stream=True,
        show=False, save=True, imgsz=768, save_txt=True,
        line_thickness=1,  # save_conf=True, #save_crop=True,
        tracker=tracker_cfg,  # botsort.yaml
        iou=0.7, conf=0.5, exist_ok=True
    )

    return results


def track2csv(tracker_cfg, model, video_in, csv_out_auto, csv_out_human, len_const):
    """
    Process YOLO-generator tracking output: ids
    and masks. Calculate mask centroids for all
    objects on the frame and centroids distance graph
    Save results in .csv files
    :param tracker_cfg: config file for tracker
    :param model: path to model.pt weights file
    :param video_in: src video file
    :param csv_out_auto: name for csv file for further automatic analysis
    :param csv_out_human: ame for csv file for further manual analysis
    :param len_const: zebra length constant for certain video file
    :return: number of frames processed
    """

    results = yolo_track(tracker_cfg, model, video_in)
    frm, all_xc, all_yc, all_idxs, all_xydists = [], [], [], [], []

    for frm_no, r in enumerate(results, 0):
        xc, yc, idxs, xydists = [], [], [], []
        if r.boxes is not None and r.boxes.id is not None:
            ids, bbs = r.boxes.id, r.boxes.xywh
            #             print(r.boxes.is_track)
            #             print(r.boxes.conf)
            for i, m in zip(ids, bbs):
                coords = np.round(m).cpu().numpy().astype(int)
                xc.append(str(coords[0]))
                yc.append(str(coords[1]))
                idxs.append(str(int(i)))

            for x, y in zip(it.combinations(xc, 2), it.combinations(yc, 2)):
                xdsq = (int(x[0]) - int(x[1])) ** 2
                ydsq = (int(y[0]) - int(y[1])) ** 2
                xydist = round((xdsq + ydsq) ** 0.5) / len_const
                xydists.append(f'{xydist:.2f}')
        else:
            idxs.append('')
            xc.append('')
            yc.append('')
            xydists.append('')

        sp = ', ,'

        with open(os.getcwd() + '/' + csv_out_auto + '.csv', 'a') as f:
            f.write(
                str(frm_no) + sp + ','.join(idxs) + sp + ','.join(xc) +
                sp + ','.join(yc) + sp + ','.join(xydists) + '\n'
            )

        frm.append(frm_no)
        all_idxs.append(idxs)
        all_xc.append(xc)
        all_yc.append(yc)
        all_xydists.append(xydists)

    data = pd.DataFrame(
        {
            'frame': frm,
            'IDs': all_idxs,
            'Xc': all_xc,
            'Yc': all_yc,
            'Dists': all_xydists
        }
    )

    data.to_csv(csv_out_human + '.csv', index=False)

    return f'{frm_no} video frames processed'


# z1 (1-st video) & z2 (2-nd video) zebra lengths to factor distances
z1_len = round(((396 - 389) ** 2 + (671 - 689) ** 2) ** 0.5)  # low-right zebra on 1-st frame
z2_len = round(((636 - 624) ** 2 + (582 - 540) ** 2) ** 0.5)  # low-right zebra on 1-st frame


def drawing(video_in, csv_in, video_out, img_out):
    """
    Parse .csv with ids and centroids coords, draw them
    on the orig video and add distance lines
    :param video_in: src video file
    :param csv_in: src csv file generated on previous step
    :param video_out: name for result video file
    :param img_out: folder name for result frames (disabled by comment)
    :return:
    """

    cap = cv2.VideoCapture(video_in)
    rate, (w, h) = 24, (int(cap.get(3)), int(cap.get(4)))  # int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # *'mp4v'
    wrt = cv2.VideoWriter(video_out + '.mp4', fourcc, rate, (w, h))

    pic_font = cv2.FONT_HERSHEY_SIMPLEX
    pic_line = cv2.LINE_AA

    if not os.path.exists(os.getcwd() + img_out):
        os.mkdir(os.getcwd() + img_out)

    with open(os.path.join(os.getcwd(), csv_in + '.csv'), 'r') as f:
        lines = f.readlines()

        while cap.isOpened():
            for line in lines:
                frm_no = line.split(' ')[0][:-1]
                ids = line.split(' ')[1].split(',')[1:-1]
                xs = line.split(' ')[2].split(',')[1:-1]
                ys = line.split(' ')[3].split(',')[1:-1]
                dists = line.split(' ')[4].split(',')[1:-1]

                ret, frame = cap.read()

                if frame is not None:
                    for cx, cy, idx in zip(xs, ys, ids):
                        if idx:
                            cv2.circle(frame, (int(cx), int(cy)), 2, (0, 0, 255), 3)
                            cv2.putText(
                                frame, idx, (int(cx) + 15, int(cy) + 15),
                                pic_font, 0.5, (0, 128, 128), 2
                            )

                    cv2.imwrite((os.getcwd() + img_out + frm_no + '.jpg'), frame)
                    wrt.write(frame)
                else:
                    pass

            wrt.release()
            cap.release()

    return 'OK'


# Main loop
if __name__ == '__main__':
    train("yolov8m.pt",  # model of middle size
          "zebras-2.yaml"
          )
    print('train DONE')

    test("runs/detect/train/weights/best.pt",  # default YOLO path to best weights
         "z2_frames_960x760_test"
         )
    print('test DONE')

    track2csv("bytetrack-2.yaml",
              "runs/detect/train/weights/best.pt",  # default YOLO path to best weights
              "15s.mp4",
              "distances_z2_a",
              "distances_z2_h",
              z1_len
              )
    print('track2csv DONE')

    drawing("15s.mp4",
            "distances_z2_a",
            "z2_all_out",
            "/out/"
            )
    print('drawing DONE')
