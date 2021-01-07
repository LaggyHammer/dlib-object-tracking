# coding: utf-8
# =====================================================================
#  Filename:    multi_object_tracking.py
#
#  py Ver:      python 3.6 or later
#
#  Description: Detects objects & tracks them through a video feed or the web-cam. Uses a pre-trained detection model.
#
#  Usage: python multi_object_tracking.py --prototxt model\MobileNetSSD_deploy.prototxt \
#         --model model\MobileNetSSD_deploy.caffemodel --label person
#         or
#        python multi_object_tracking.py --prototxt model\MobileNetSSD_deploy.prototxt \
#        --model model\MobileNetSSD_deploy.caffemodel --label person --out output.avi
#         or
#        python multi_object_tracking.py --prototxt model\MobileNetSSD_deploy.prototxt \
#        --model model\MobileNetSSD_deploy.caffemodel --video test.mp4 --label person --out output.avi
#
#  Note: Requires opencv 3.4.2 or later
#
#  Author: Ankit Saxena (ankch24@gmail.com)
# =====================================================================

import numpy as np
import argparse
import imutils
import cv2
import dlib
import multiprocessing
from imutils.video import FPS
from utils import update_tracker, create_videowriter, forward_passer


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', type=str, required=True,
                    help='path to detection model')
    ap.add_argument('-p', '--prototxt', type=str, required=True,
                    help='path to prototxt file')
    ap.add_argument('-c', '--min_confidence', type=float, default=0.2,
                    help='minimum confidence for a detection')
    ap.add_argument('-v', '--video', type=str,
                    help='path to optional video file')
    ap.add_argument('-l', '--label', type=str, required=True,
                    help='item to detect & track')
    ap.add_argument('-o', '--output', type=str,
                    help='item to detect & track')
    arguments = vars(ap.parse_args())

    return arguments


def start_tracker(box, label, frame, in_queue, out_queue):
    """
    Starts a tracker and then continuously pushes its updates to a queue
    :param box: detection box to be tracked
    :param label: label for the detection
    :param frame: frame to place the detector on
    :param in_queue: incoming queue of frames
    :param out_queue: queue to push the labelled trackers on
    """

    # initiate tracker
    tracker = dlib.correlation_tracker()
    rectangle = dlib.rectangle(box[0], box[1], box[2], box[3])
    tracker.start_track(frame, rectangle)

    while True:
        # get frame
        frame = in_queue.get()

        if frame is not None:
            # push the tracker after updating it
            out_queue.put((label, update_tracker(tracker, frame)))


def main(classes, proto, model, video, label_input, output, min_confidence):

    in_queues = []
    out_queues = []

    # pre-load detection model
    print("[INFO] loading detection model...")
    net = cv2.dnn.readNetFromCaffe(prototxt=proto, caffeModel=model)

    print('[INFO] Starting video stream...')
    if not video:
        # start web-cam feed
        vs = cv2.VideoCapture(0)

    else:
        # start video stream
        vs = cv2.VideoCapture(video)

    # initializing variables
    writer = None

    fps = FPS().start()

    # main loop
    while True:

        grabbed, frame = vs.read()

        if frame is None:
            break

        # resize the frame & convert to RGB color space (dlib needs RGB)
        frame = imutils.resize(frame, width=600)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if output is not None and writer is None:
            # initialize output file writer
            writer = create_videowriter(output, 30, (frame.shape[1], frame.shape[0]))

        # if no item in the input queue
        if len(in_queues) == 0:
            height, width = frame.shape[:2]
            # get detections from the model
            detections = forward_passer(net, frame, timing=False)

            # loop through all detections
            for i in np.arange(0, detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                if confidence > min_confidence:
                    index = int(detections[0, 0, i, 1])
                    label = classes[index]

                    if label != label_input:
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    bound_box = box.astype('int')

                    # initiate multiprocessing queues
                    in_q = multiprocessing.Queue()
                    out_q = multiprocessing.Queue()
                    in_queues.append(in_q)
                    out_queues.append(out_q)

                    # initiating daemon process
                    p = multiprocessing.Process(target=start_tracker,
                                                args=(bound_box, label, rgb, in_q, out_q))
                    p.daemon = True
                    p.start()

                    # draw rectangles around the tracked detections
                    cv2.rectangle(frame, (bound_box[0], bound_box[1]), (bound_box[2], bound_box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, label, (bound_box[0], bound_box[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 255, 0), 2)

        # if detections already identified
        else:

            # push frames
            for in_q in in_queues:
                in_q.put(rgb)

            # get labelled boxes
            for out_q in out_queues:
                label, label_box = out_q.get()

                # draw rectangles around the tracked detections
                cv2.rectangle(frame, (label_box[0], label_box[1]), (label_box[2], label_box[3]),
                              (0, 255, 0), 2)
                cv2.putText(frame, label, (label_box[0], label_box[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 0), 2)

        if writer is not None:
            writer.write(frame)

        # show result
        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(1) & 0xFF

        # quit if 'q' is pressed
        if key == ord('q'):
            break

        fps.update()

    fps.stop()
    print(f'[INFO] Elapsed time: {round(fps.elapsed(), 2)}')
    print(f'[INFO] approximate FPS: {round(fps.fps(), 2)}')

    # release video writer end-point
    if writer is not None:
        writer.release()

    # release video stream end-point
    cv2.destroyAllWindows()
    vs.release()


def web_main(classes, proto, model, video, label_input, output, min_confidence):
    in_queues = []
    out_queues = []

    # pre-load detection model
    print("[INFO] loading detection model...")
    net = cv2.dnn.readNetFromCaffe(prototxt=proto, caffeModel=model)

    print('[INFO] Starting video stream...')

    # start video stream
    vs = video

    # main loop
    while True:

        grabbed, frame = vs.read()

        if frame is None:
            break

        # resize the frame & convert to RGB color space (dlib needs RGB)
        frame = imutils.resize(frame, width=600)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if no item in the input queue
        if len(in_queues) == 0:
            height, width = frame.shape[:2]
            # get detections from the model
            detections = forward_passer(net, frame, timing=False)

            # loop through all detections
            for i in np.arange(0, detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                if confidence > min_confidence:
                    index = int(detections[0, 0, i, 1])
                    label = classes[index]

                    if label != label_input:
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    bound_box = box.astype('int')

                    # initiate multiprocessing queues
                    in_q = multiprocessing.Queue()
                    out_q = multiprocessing.Queue()
                    in_queues.append(in_q)
                    out_queues.append(out_q)

                    # initiating daemon process
                    p = multiprocessing.Process(target=start_tracker,
                                                args=(bound_box, label, rgb, in_q, out_q))
                    p.daemon = True
                    p.start()

                    # draw rectangles around the tracked detections
                    cv2.rectangle(frame, (bound_box[0], bound_box[1]), (bound_box[2], bound_box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, label, (bound_box[0], bound_box[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 255, 0), 2)

        # if detections already identified
        else:

            # push frames
            for in_q in in_queues:
                in_q.put(rgb)

            # get labelled boxes
            for out_q in out_queues:
                label, label_box = out_q.get()

                # draw rectangles around the tracked detections
                cv2.rectangle(frame, (label_box[0], label_box[1]), (label_box[2], label_box[3]),
                              (0, 255, 0), 2)
                cv2.putText(frame, label, (label_box[0], label_box[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 0), 2)

        # show result
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


if __name__ == '__main__':

    args = get_arguments()

    # classes that the model can recognize // change according to the model
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    main(classes=CLASSES, proto=args['prototxt'], model=args['model'], video=args.get('video', False),
         label_input=args['label'], output=args.get('output', None), min_confidence=args['min_confidence'])
