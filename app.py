from flask import Flask, render_template, request
from PIL import Image
from ultralytics import YOLO
import os
import shutil
import cv2
from datetime import datetime
import glob
import uuid

app = Flask(__name__)
app.config['DEBUG']=True

UPLOAD_FOLDER = 'static/uploads'
DETECT_FOLDER = 'static/detections'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECT_FOLDER'] = DETECT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)

model = YOLO('best.pt')  # Model loaded here (cold start)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part in the request.'
    file = request.files['file']
    if file.filename == '':
        return 'No file selected.'

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + '_' + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    results = model.predict(filepath)
    for result in results:
        im_array = result.plot()
        output_path = os.path.join(app.config['DETECT_FOLDER'], 'result_' + filename)
        cv2.imwrite(output_path, im_array)

    return render_template('result.html', result_image='/' + output_path)

# NO app.run() here — Gunicorn will run the app