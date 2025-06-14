from flask import Flask, render_template, request
from PIL import Image
from ultralytics import YOLO
import os
import shutil
from datetime import datetime
import glob
import uuid

print("🔍 Current working directory:", os.getcwd())
print("📂 Files in current directory:", os.listdir())
print("📂 Files in 'static' directory:", os.listdir('static'))
print("📂 Files in 'templates' directory:", os.listdir('templates'))

if not os.path.isfile('best.pt'):
    print("❌ best.pt NOT FOUND!")
else:
    print("✅ best.pt FOUND!")
# 🔍 Debug code ends here


app = Flask(__name__)

model = YOLO('best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    detect_dir = os.path.join('runs', 'detect')
    
    # Clear old detection results if they exist
    if os.path.exists(detect_dir):
        shutil.rmtree(detect_dir)

    if 'image' not in request.files:
        return "No file uploaded", 400

    img_file = request.files['image']

    # Save the uploaded image temporarily
    img_path = os.path.join('static', img_file.filename)
    img_file.save(img_path)

    # Run YOLO model and force output into 'runs/detect'
    results = model.predict(img_path, save=True, project='runs', name='detect')

    if not os.path.exists(detect_dir):
        return "Error: Detection output not found.", 500
    
    subdirs = [os.path.join(detect_dir, d) for d in os.listdir(detect_dir) if os.path.isdir(os.path.join(detect_dir, d))]

    if not subdirs:
        return "Error: No detection output found.", 500

    latest_dir = max(subdirs, key=os.path.getmtime)
    # Find the first detected image (jpg or png)
    detected_img_name = next((f for f in os.listdir(latest_dir) if f.endswith(('.jpg', '.png'))), None)

    if not detected_img_name:
        return "Error: No image found in YOLO output folder."

    detected_img_path = os.path.join(latest_dir, detected_img_name)
    final_path = os.path.join('static', 'result_{}.jpg'.format(uuid.uuid4()))

    shutil.copyfile(detected_img_path, final_path)

    return render_template('result.html', result_image=final_path)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Get port from Render
    app.run(host='0.0.0.0', port=port)