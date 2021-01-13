# dlib-object-tracking
Object tracking using a pre-trained object detection model & the dlib correlation tracker.

The algorithm starts streaming from a video file or a live web-cam feed.
Object detection through a pre-trained model is performed on the feed till a positive for the input label is found.

The correlation tracker used here uses discriminative correlation filters to localize the target,
building upon a Minimum Output Sum of Squared Error (MOSSE) filter. This makes the tracker robust to
variations in lighting, pose & scale.

More information about the tracking algorithm can be found below: <br>
1. [Object Tracking in OpenCV](http://blog.dlib.net/2015/02/dlib-1813-released.html)
2. [Visual Object Tracking using Adaptive Correlation Filters](https://www.cs.colostate.edu/~vision/publications/bolme_cvpr10.pdf)
3. [Accurate Scale Estimation for Robust Visual Tracking](http://www.bmva.org/bmvc/2014/papers/paper038/index.html)

## Single Object Tracking
The detection result is taken to be as the single object with the best detection confidence.

A tracker is subsequently placed on the video stream and is updated every frame to track the object.

### Usage
```commandline
python single_object_tracking.py --prototxt model\MobileNetSSD_deploy.prototxt --model model\MobileNetSSD_deploy.caffemodel --label person
python single_object_tracking.py --prototxt model\MobileNetSSD_deploy.prototxt --model model\MobileNetSSD_deploy.caffemodel --label person --out output.avi
python single_object_tracking.py --prototxt model\MobileNetSSD_deploy.prototxt --model model\MobileNetSSD_deploy.caffemodel --video test.mp4 --label person --out output.avi
```

## Multiple Object Tracking
All the detection results for the input label are tracked to provide multiple detections.
 
The tracking processes are distributed via python's multiprocessing module.

### Usage
```commandline
python multi_object_tracking.py --prototxt model\MobileNetSSD_deploy.prototxt --model model\MobileNetSSD_deploy.caffemodel --label person
python multi_object_tracking.py --prototxt model\MobileNetSSD_deploy.prototxt --model model\MobileNetSSD_deploy.caffemodel --label person --out output.avi
python multi_object_tracking.py --prototxt model\MobileNetSSD_deploy.prototxt --model model\MobileNetSSD_deploy.caffemodel --video test.mp4 --label person --out output.avi
```