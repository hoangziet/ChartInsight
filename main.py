from flask import Flask, render_template, request
import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import cls, bars, pies

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CHARTS = ["Bar", "Line", "Pie"]

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            predicted_class = cls.classify(filepath)
            result = CHARTS[predicted_class]
            
            if result == 'Pie':
                pie_info = pies.to_dataframe(filepath)
                char_info = pie_info.to_dict(orient='records') 
            elif result == 'Bar':
                bar_info = bars.to_dataframe(filepath)
                char_info = bar_info[['label', 'color', 'value']].to_dict('records')   
            else:
                char_info= False
           
        return render_template('index.html', filename=file.filename, result=result, char_info=char_info)
    return render_template('index.html', filename=None, result=None)

if __name__ == '__main__':
    app.run(debug=True, port = 5001)
