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

RESULT_FOLDER = 'static/results'
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Load YOLOv8 model (only once)
model = YOLO('best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part in the request'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(RESULT_FOLDER, filename)
        file.save(file_path)

        # Run YOLO model on the saved file
        results = model(file_path)
        
        # Save the result image as result.jpg (overwrite each time)
        results[0].save(filename=os.path.join(RESULT_FOLDER, 'result.jpg'))

        # Redirect to result page
        return redirect(url_for('result'))

@app.route('/result')
def result():
    result_path = os.path.join(RESULT_FOLDER, 'result.jpg')
    return render_template('result.html', result_image=result_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)