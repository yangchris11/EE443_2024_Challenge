# visualize tracking results from tracking/IoU_Tracker.py

import os
import os.path as osp
import sys
import time

import cv2

import numpy as np

results = "/media/cycyang/sda1/EE443_final/runs/tracking/inference/camera_0008.txt"
image_path = "/media/cycyang/sda1/EE443_final/data/test/camera_0008"

# load the tracking results
tracking_results = np.loadtxt(results, delimiter=',', dtype=None)

# get the unique frame IDs
frame_ids = np.unique(tracking_results[:, 2])

# group the tracking results by frame ID
tracking_results = [tracking_results[tracking_results[:, 2] == frame_id] for frame_id in frame_ids]

for frame_id, tracking_result in zip(frame_ids, tracking_results):
    # pad the frame_id with zeros to 5
    frame_id = str(int(float(frame_id))).zfill(5)
    img_path = osp.join(image_path, frame_id + '.jpg')
    print(f"Visualizing frame {frame_id} from {img_path}")

    new_img_path = osp.join("/media/cycyang/sda1/EE443_final/runs/tracking/inference/vis", frame_id + '.jpg')


    img = cv2.imread(img_path)
    for track in tracking_result:
        x, y, w, h = map(int, track[3:7])
        track_id = int(track[1])
        print(f"Draw bounding box at ({x}, {y}, {w}, {h}) with track id {track_id}")
        # draw bounding box with track id
        # you can use cv2.rectangle and cv2.putText to draw the bounding box and track id
        # remember to save the image after drawing the bounding box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, str(track_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(new_img_path, img)