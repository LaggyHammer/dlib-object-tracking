import os
from flask import Flask, Response, render_template, redirect, url_for, flash, request
import cv2
from multi_object_tracking import web_main

app = Flask(__name__)
video = cv2.VideoCapture(0)
label = "person"

# classes that the model can recognize // change according to the model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'mp4'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def default():
    return redirect(url_for('index'))


@app.route('/home/')
def index():
    global video
    global label
    video = cv2.VideoCapture(r"static\videos\r6s_k.mp4")
    label = "person"
    return render_template('upload_index.html')


@app.route('/cat/')
def cat():
    global video
    global label
    video = cv2.VideoCapture(r"static\videos\cat.mp4")
    label = "cat"
    return render_template('index.html')


@app.route('/race/')
def race():
    global video
    global label
    video = cv2.VideoCapture(r"static\videos\race.mp4")
    label = "person"
    return render_template('index.html')


@app.route('/car/')
def car():
    global video
    global label
    video = cv2.VideoCapture(r"static\videos\race3.mp4")
    label = "car"
    return render_template('index.html')


@app.route('/person/')
def person():
    global video
    global label
    video = cv2.VideoCapture(r"static\videos\r6s_k.mp4")
    label = "person"
    return render_template('index.html')


@app.route('/live/')
def live():
    global video
    global label
    video = cv2.VideoCapture(0)
    label = "person"
    return render_template('index.html')


def gen(feed):
    while True:
        success, image = feed.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route("/explain/")
def explain():
    return render_template('how_it_works.html')


@app.route('/video_feed/')
def video_feed():
    global video
    global label
    return Response(web_main(classes=CLASSES, proto=r"model\MobileNetSSD_deploy.prototxt",
                             model=r"model\MobileNetSSD_deploy.caffemodel",
                             video=video,
                             label_input=label, output=None,
                             min_confidence=0.2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload/')
def upload_form():
    return render_template('upload_index.html')


@app.route('/upload/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No video selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Video successfully uploaded and displayed')
        global video
        global label
        video = cv2.VideoCapture(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        label = "person"
        return render_template('index.html')
    else:
        flash('Allowed image types are -> mp4')
        return redirect(request.url)


if __name__ == '__main__':
    app.run()
