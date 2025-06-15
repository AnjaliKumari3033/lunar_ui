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

model = YOLO('best.pt')  # Replace 'best.pt' with your actual model path if needed

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Image upload and processing route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part in the request."

    file = request.files['image']
    if file.filename == '':
        return "No file selected."

    # Generate unique folder for this request
    request_id = str(uuid.uuid4())
    request_upload_folder = os.path.join(UPLOAD_FOLDER, request_id)
    request_result_folder = os.path.join(RESULT_FOLDER, request_id)

    os.makedirs(request_upload_folder, exist_ok=True)
    os.makedirs(request_result_folder, exist_ok=True)

    # Save original image
    img_path = os.path.join(request_upload_folder, file.filename)
    file.save(img_path)

    # Run YOLO inference
    results = model(img_path)

    # Save result image to result folder
    for r in results:
        im_array = r.plot()  # Plot result with boxes
        im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB
        result_img_path = os.path.join(request_result_folder, 'result.jpg')
        im.save(result_img_path)

    # After processing, cleanup uploaded image folder (optional: keep if you want)
    shutil.rmtree(request_upload_folder)
     # Show result
    result_img_rel_path = os.path.join(request_result_folder, 'result.jpg')
    return render_template('result.html', result_image=result_img_rel_path)

# To serve result images
@app.route('/static/results/<path:filename>')
def result_file(filename):
    return send_from_directory('static/results', filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)