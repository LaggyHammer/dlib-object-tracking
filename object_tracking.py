import numpy as np
import argparse
import imutils
import cv2
import dlib
from imutils.video import FPS


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


if __name__ == '__main__':
    args = get_arguments()

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    print("[INFO] loading detection model...")
    net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

    print('[INFO] Starting video stream...')
    vs = cv2.VideoCapture(args['video'])
    tracker = None
    writer = None
    label = ""

    fps = FPS().start()

    while True:
        grabbed, frame = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=600)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if args['output'] is not None and writer is None:

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(args['output'], fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

        if tracker is None:

            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)

            net.setInput(blob)
            detections = net.forward()

            if len(detections) > 0:

                i = np.argmax(detections[0, 0, :, 2])

                conf = detections[0, 0, i, 2]
                label = CLASSES[int(detections[0, 0, i, 1])]

                if conf > args["min_confidence"] and label == args["label"]:

                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    start_x, start_y, end_x, end_y = box.astype('int')

                    tracker = dlib.correlation_tracker()
                    rectangle = dlib.rectangle(start_x, start_y, end_x, end_y)
                    tracker.start_track(rgb, rectangle)

                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    cv2.putText(frame, label, (start_x, start_y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        else:
            tracker.update(rgb)
            position = tracker.get_position()

            start_x = int(position.left())
            start_y = int(position.top())
            end_x = int(position.right())
            end_y = int(position.bottom())

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.putText(frame, label, (start_x, start_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        fps.update()

    fps.stop()
    print(f'[INFO] Elapsed time: {round(fps.elapsed(), 2)}')
    print(f'[INFO] approximate FPS: {round(fps.fps(), 2)}')

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()
    vs.release()