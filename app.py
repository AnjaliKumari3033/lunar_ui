from flask import Flask, render_template, request
from PIL import Image
from ultralytics import YOLO
import os
import shutil
from datetime import datetime
import glob
import uuid

app = Flask(__name__)

model = YOLO('best.pt')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    detect_dir = os.path.join('runs', 'detect')
    shutil.rmtree(detect_dir)
    os.makedirs(detect_dir)

    if 'image' not in request.files:
        return "No file uploaded", 400

    img_file = request.files['image']
    
    # Save the uploaded image temporarily
    img_path = os.path.join('static', img_file.filename)
    img_file.save(img_path)

    results = model.predict(img_path, save=True)
    
    detect_dir = os.path.join('runs', 'detect')
    subdirs = [os.path.join(detect_dir, d) for d in os.listdir(detect_dir) if os.path.isdir(os.path.join(detect_dir, d))]
    latest_dir = max(subdirs, key=os.path.getmtime)

    # Find first image file in output (e.g., jpg, png)
    detected_img_name = next((f for f in os.listdir(latest_dir) if f.endswith(('.jpg', '.png'))), None)

    if not detected_img_name:
        return "Error: No image found in YOLO output folder."

    detected_img_path = os.path.join(latest_dir, detected_img_name)
    final_path = os.path.join('static', 'result_{}.jpg'.format(uuid.uuid4()))

    shutil.copyfile(detected_img_path, final_path)

   
    return render_template('result.html', result_image=f'{final_path}')
   
if __name__ == '__main__':
    app.run(debug=True)