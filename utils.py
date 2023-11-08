# ---UTILS MODULE---

import numpy as np
import pandas as pd
import cv2
import os
import shutil
import natsort as ns
import json


def get_video_specs(path):
    """
    Returns video specifications (width, height, frame rate)
    for input selected path
    :param path: path to input
    :return: int
    """

    cap = cv2.VideoCapture(path)

    width = cap.get(3)
    height = cap.get(4)
    frm_rate = cap.get(5)

    return int(width), int(height), int(frm_rate)


def video2frames(src, out, sample, xr, yr, x, y, w, h):
    """
    Sampling frames from video with constant frequency

    src: path to input video
    out: path to result
    sample: write each "sample-th" frame
    xr, yr - resize to new resolution (abs values)
    x, y, w, h - pars to cutting each frame
    """

    cap = cv2.VideoCapture(src)

    # Check if cap opened successfully
    if not cap.isOpened():
        print('Error opening video')

    if not os.path.exists(os.getcwd() + '/' + out):
        os.mkdir(os.getcwd() + '/' + out)

    i, s = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            if i % sample == 0:  # write each 'sample-th' frame
                frame = cv2.resize(frame, (xr, yr), cv2.INTER_NEAREST)
                # cv2.rectangle(frame , (y, x), (y+h, x+w), (255, 255, 255), thickness=2)
                frame = frame[x:w, y:h]
                cv2.imwrite(os.getcwd() + '/' + out + '/' + 'z1-' + str(i) + '.jpg', frame)

                s += 1
            i += 1
        else:
            break
    cap.release()

    return f'OK, {s} frames were saved'


def frames2video(src, out, rate):

    """
    Assembling video from frames set.
    Assumed all frames of the same size.

    src: path to input video
    out: path to result
    rate: frame rate of output
    """

    names = ns.natsorted(os.listdir(os.getcwd() + '/' + src))
    frame_1 = cv2.imread(os.getcwd() + '/' + src + '/' + names[0])
    h, w = frame_1.shape[0], frame_1.shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # *'mp4v'
    wrt = cv2.VideoWriter(out, fourcc, rate, (w, h))

    i = 0
    for name in names:
        frame = cv2.imread(os.getcwd() + '/' + src + '/' + name)
        wrt.write(frame)
        i += 1

    wrt.release()

    return f'OK, {i} frames were assembled in video'


def click_sampler(name, size, out):
    """
    Sampler fn based on OpenCV callbacks with mouse clicks
    Sampling tiles of specified size from a series of images
    by defining its center via mouse RClick.
    :param name: src folder name
    :param size: cropped tile size
    :param out: path to results folder
    :return: str
    """

    if not os.path.exists(out):
        os.mkdir(out)

    if size <= 0:
        raise ValueError('tile size should be > 0')

    img = cv2.imread(name)
    img_c = img.copy()

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.resizeWindow(name, 1000, 1000)

    coords = []

    def mouse_click(event, x, y, flags, param):

        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
            cv2.imshow(name, img)
            coords.append((x, y))

        return coords

    cv2.setMouseCallback(name, mouse_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    t = 0
    nm, ext = name.split('.')[0], '.' + name.split('.')[1]

    for coord in coords:
        x, y = coord[0], coord[1]
        x0, y0 = x - int(size / 2), y - int(size / 2)
        xw, yh = x + int(size / 2), y + int(size / 2)

        if x0 < 0:
            x0 = 0
        if xw > img.shape[1]:
            xw = img.shape[1]
        if y0 < 0:
            y0 = 0
        if yh > img.shape[0]:
            yh = img.shape[0]

        tile = img_c[y0:yh, x0:xw]

        if tile.shape[0] < size and tile.shape[1] < size:
            v_pad = np.ones([size - tile.shape[0], tile.shape[1]]) * 127
            v_pad_3 = np.stack([v_pad, v_pad, v_pad], -1)
            tile = np.vstack([tile, v_pad_3])
            h_pad = np.ones([tile.shape[0], size - tile.shape[1]]) * 127
            h_pad_3 = np.stack([h_pad, h_pad, h_pad], -1)
            tile = np.hstack([tile, h_pad_3])

        if tile.shape[0] < size:
            v_pad = np.ones([size - tile.shape[0], size]) * 127
            v_pad_3 = np.stack([v_pad, v_pad, v_pad], -1)
            tile = np.vstack([tile, v_pad_3])

        if tile.shape[1] < size:
            h_pad = np.ones([size, size - tile.shape[1]]) * 127
            h_pad_3 = np.stack([h_pad, h_pad, h_pad], -1)
            tile = np.hstack([tile, h_pad_3])

        cv2.imwrite(out + '/' + os.path.basename(nm) + '_' + str(x) + '_' + str(y) + ext, tile)

        t += 1

    return f'{t} tiles were generated'


def VIA2YOLO_detect(data, cls_codes, out_dir, imgsize):
    """
    Converts VIA detection (x,y,w,h-rects expected) annotations (one .csv file)
    to individual YOLO-formatted txt files for each img with objects.
    Empty images have no any files. Params:

    -data (path-like string) - source .csv  VIA annotation
    -cls_codes (dict) - class name-to-index mapping: {'dog':0, 'cat':1 ...}
    -out - dir for generated .txt files (will be created if not exist)
    -imgsize (tuple of ints) - size (H,W) of annotated images (expected all the same)
    """

    annot = pd.read_csv(data, index_col=0)
    annot.drop(['file_size', 'file_attributes'], axis=1, inplace=True)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for label, row in annot.iterrows():
        name, r_count, coords, cls = label, int(row[0]), json.loads(row[2]), str(json.loads(row[3]))

        #         # with empty files for empty pics
        if r_count == 0:
            with open(out_dir + '/' + str(label[:-4]) + '.txt', 'w') as f:
                f.write('')

        if r_count == 1:
            if coords['name'] == 'rect':
                xywh = []
                xywh.append(coords['x'] / imgsize[0] + (coords['width'] / imgsize[0]) / 2)
                xywh.append(coords['y'] / imgsize[1] + (coords['height'] / imgsize[0]) / 2)
                xywh.append(coords['width'] / imgsize[0])
                xywh.append(coords['height'] / imgsize[1])

                for key in cls_codes.keys():
                    if key in cls:
                        cls_id = cls_codes[key]
                    else:
                        cls_id = 'unknown'

                fin_str = str(cls_id) + ' ' + str(xywh)[1:-1].replace(',', '')

                with open(out_dir + '/' + str(label[:-4]) + '.txt', 'w') as f:
                    f.write(fin_str + '\n')

        if r_count > 1:
            if coords['name'] == 'rect':
                xywh = []
                xywh.append(coords['x'] / imgsize[0] + (coords['width'] / imgsize[0]) / 2)
                xywh.append(coords['y'] / imgsize[1] + (coords['height'] / imgsize[0]) / 2)
                xywh.append(coords['width'] / imgsize[0])
                xywh.append(coords['height'] / imgsize[1])

                for key in cls_codes.keys():
                    if key in cls:
                        cls_id = cls_codes[key]
                    else:
                        cls_id = 'unknown'

                fin_str = str(cls_id) + ' ' + str(xywh)[1:-1].replace(',', '')

                with open(out_dir + '/' + str(label[:-4]) + '.txt', 'a') as f:
                    f.write(fin_str + '\n')

    return annot['region_attributes'].value_counts()


def labels4images(root, img_prefix, lab_prefix):
    """
    Accepts images and labels dirs in root dir, search
    names in images train and val dirs (already created
    by user) and divides labels accordingly
    """

    img_names = os.listdir(root + '/' + img_prefix + '/train/')
    lab_names = os.listdir(root + '/' + lab_prefix)

    if not os.path.exists(root + '/' + lab_prefix + '/train'):
        os.mkdir(root + '/' + lab_prefix + '/train')
    if not os.path.exists(root + '/' + lab_prefix + '/val'):
        os.mkdir(root + '/' + lab_prefix + '/val')

    lnames, inames = [], []

    for f in lab_names:
        name, ext = f.split('.')[0], f.split('.')[1]
        lnames.append(name)
    for f in img_names:
        name = f[:-4]
        inames.append(name)

    for name in lnames:
        if name in inames:
            shutil.move(
                root + '/' + lab_prefix + '/' + name + '.' + ext,
                root + '/' + lab_prefix + '/train/' + name + '.' + ext
            )
        else:
            shutil.move(
                root + '/' + lab_prefix + '/' + name + '.' + ext,
                root + '/' + lab_prefix + '/val/' + name + '.' + ext
            )

    return 'OK'
