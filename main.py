from flask import Flask, render_template, request
import os
import cv2

# utils for classification, bar and pie helper modules
from utils import cls, bars, pies

CHARTS = ["Bar", "Line", "Pie"]
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            return "No selected file", 400
        if not allowed_file(file.filename):
            return "Invalid file format. Please upload an image (png, jpg, jpeg, bmp).", 400

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

        if not char_info:
            char_info = "No chart information extracted."
           
        return render_template('index.html', filename=file.filename, result=result, char_info=char_info)
    return render_template('index.html', filename=None, result=None)

if __name__ == '__main__':
    app.run(debug=True, port = 5002)
    # CUDA_VISIBLE_DEVICES = -1 python main.py