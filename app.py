from flask import Flask, render_template, request
from PIL import Image
from ultralytics import YOLO
import os
import shutil
from datetime import datetime
import glob
import uuid

app = Flask(__name__)
app.config['DEBUG']=True

model_path = 'best.pt'
if not os.path.isfile(model_path):
    raise FileNotFoundError("YOLO model file 'best.pt' not found!")
model = YOLO(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    detect_dir = os.path.join('runs', 'detect')
    static_dir = 'static'
    
    # ✅ 1. Clear old detection results (clean up YOLO output)
    if os.path.exists(detect_dir):
        shutil.rmtree(detect_dir)

    # ✅ 2. Clean old result and upload images from static
    for filename in os.listdir(static_dir):
        if filename.startswith('result_') or filename.startswith('upload_'):
            os.remove(os.path.join(static_dir, filename))

    if 'image' not in request.files:
        return "No file uploaded", 400

    img_file = request.files['image']
    if img_file.filename == '':
        return "Error: Empty filename.", 400

    # ✅ 3. Save uploaded image with a unique name
    unique_filename = f"upload_{uuid.uuid4()}.jpg"
    upload_path = os.path.join(static_dir, unique_filename)
    img_file.save(upload_path)

    # ✅ 4. Run YOLO model
    results = model.predict(upload_path, save=True, project='runs', name='detect', exist_ok=True)

    # ✅ 5. Locate the latest YOLO output folder
    if not os.path.exists(detect_dir):
        return "Error: Detection output not found.", 500

    detect_subdirs = [os.path.join(detect_dir, d) for d in os.listdir(detect_dir) if os.path.isdir(os.path.join(detect_dir, d))]
    if not detect_subdirs:
         return "Error: No detection output found.", 500

    latest_dir = max(detect_subdirs, key=os.path.getmtime)
    detected_imgs = [f for f in os.listdir(latest_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not detected_imgs:
        return "Error: YOLO did not produce any image output.", 500

    # ✅ 6. Copy YOLO result image to static/ with unique name
    detected_img_path = os.path.join(latest_dir, detected_imgs[0])
    final_result_path = os.path.join(static_dir, f"result_{uuid.uuid4()}.jpg")
    shutil.copyfile(detected_img_path, final_result_path)

    # ✅ 7. Remove uploaded image to free space
    os.remove(upload_path)

    return render_template('result.html', result_image=final_result_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)