from tensorflow.keras.models import load_model
import numpy as np
import cv2 
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models/classification/chart_cls.keras")

# Load classification model
model = load_model(MODEL_DIR)

def classify(image_path: str) -> int:
    """
    Classify the image by using the classification model.
    
    Args:
        image (np.ndarray): Image to classify.
        
    Returns:
        int: Predicted class.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    pred = model.predict(image)
    pred = np.argmax(pred, axis=1)[0]
    return pred



