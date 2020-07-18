# coding: utf-8
# =====================================================================
#  Filename:    utils.py
#
#  py Ver:      python 3.6 or later
#
#  Description: Utility functions for the object detection script(s)
#
#  Note: Requires opencv 3.4.2 or later
#
#  Author: Ankit Saxena (ankch24@gmail.com)
# =====================================================================

import cv2
import time


def update_tracker(tracker, frame):
    """
    Update the correlation tracker every iteration of the loop
    :param tracker: tracker to be updated
    :param frame: frame to update the tracker on
    :return: updated tracker box
    """
    tracker.update(frame)
    position = tracker.get_position()

    start_x = int(position.left())
    start_y = int(position.top())
    end_x = int(position.right())
    end_y = int(position.bottom())

    return start_x, start_y, end_x, end_y


def create_videowriter(file_name, fps, size):
    """
    Creates a video writer object to save the video feed module results
    :param file_name: file to save the video in
    :param fps: frames per second for the video
    :param size: size/resolution of the video feed
    :return: video writer object to write to
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file_name, fourcc, fps, size, True)

    return out


def forward_passer(net, image, timing=True):
    """
    Returns results from a single pass on a Deep Neural Net for a given list of layers
    :param net: Deep Neural Net
    :param image: image to do the pass on
    :param timing: show detection time or not
    :return: results obtained from the forward pass
    """
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.007843, (w, h), 127.5)
    start = time.time()
    net.setInput(blob)
    scores = net.forward()
    end = time.time()

    if timing:
        print(f"[INFO] detection in {round(end - start, 2)} seconds")

    return scores
