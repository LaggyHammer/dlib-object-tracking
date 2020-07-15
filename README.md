# dlib-object-tracking
Object tracking using a pre-trained object detection model & the dlib correlation tracker.

The algorithm starts streaming from a video file or a live web-cam feed.
Object detection through a pre-trained model is performed on the feed till a positive for the input label is found.
The detection result is taken to be as the single object with the best detection confidence.

A tracker is subsequently placed on the video stream and is updated every frame to track the object.

## Usage
```commandline
python single_object_tracking.py --prototxt model\MobileNetSSD_deploy.prototxt --model model\MobileNetSSD_deploy.caffemodel --label person
python single_object_tracking.py --prototxt model\MobileNetSSD_deploy.prototxt --model model\MobileNetSSD_deploy.caffemodel --label person --out output.avi
python single_object_tracking.py --prototxt model\MobileNetSSD_deploy.prototxt --model model\MobileNetSSD_deploy.caffemodel --video test.mp4 --label person --out output.avi
```