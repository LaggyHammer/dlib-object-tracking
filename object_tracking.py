import numpy as np
import argparse
import imutils
import cv2
import dlib
import time
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
    :param net: Deep Neural Net (usually a pre-loaded .pb file)
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


def create_tracker(detections, width, height, best_detect, frame):
    box = detections[0, 0, best_detect, 3:7] * np.array([width, height, width, height])
    start_x, start_y, end_x, end_y = box.astype('int')

    tracker = dlib.correlation_tracker()
    rectangle = dlib.rectangle(start_x, start_y, end_x, end_y)
    tracker.start_track(frame, rectangle)

    return tracker, box.astype('int')


def update_tracker(tracker, frame):
    tracker.update(frame)
    position = tracker.get_position()

    start_x = int(position.left())
    start_y = int(position.top())
    end_x = int(position.right())
    end_y = int(position.bottom())

    return start_x, start_y, end_x, end_y


def main(classes, proto, model, video, label_input, output, min_confidence):

    print("[INFO] loading detection model...")
    net = cv2.dnn.readNetFromCaffe(prototxt=proto, caffeModel=model)

    print('[INFO] Starting video stream...')
    if not video:
        vs = cv2.VideoCapture(0)

    else:
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

        if output is not None and writer is None:

            writer = create_videowriter(output, 30, (frame.shape[1], frame.shape[0]))

        if tracker is None:

            height, width = frame.shape[:2]
            detections = forward_passer(net, frame, timing=False)

            if len(detections) > 0:

                i = np.argmax(detections[0, 0, :, 2])

                confidence = detections[0, 0, i, 2]
                label = classes[int(detections[0, 0, i, 1])]

                if confidence > min_confidence and label == label_input:

                    tracker, points = create_tracker(detections, width, height, best_detect=i, frame=rgb)

                    cv2.rectangle(frame, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 2)
                    cv2.putText(frame, label, (points[0], points[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        else:

            points = update_tracker(tracker, rgb)

            cv2.rectangle(frame, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (points[0], points[1] - 15),
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


if __name__ == '__main__':
    args = get_arguments()

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    main(classes=CLASSES, proto=args['prototxt'], model=args['model'], video=args.get('video', False),
         label_input=args['label'], output=args.get('output', None), min_confidence=args['min_confidence'])
