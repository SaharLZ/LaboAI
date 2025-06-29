from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import joblib
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load models
skin_model = tf.keras.models.load_model('./model/skin_cancer_classifier_v1.keras')
breast_model = joblib.load('./model/breast_cancer_model.pkl')
lung_cancer_model = joblib.load('./model/lung_cancer_model.pkl')

skin_class_names = [
    'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma',
    'melanoma', 'nevus', 'pigmented benign keratosis',
    'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
]

selected_features_breast = [
    "perimeter_worst",
    "area_se",
    "concavity_summary",
]

selected_features_lung = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']

IMG_SIZE = (180, 180)

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.args.get('model', 'model1')

    if model_type == "model1":  
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        img = Image.open(file).convert("RGB").resize(IMG_SIZE)
        img = preprocess_input(np.array(img))
        img = np.expand_dims(img, axis=0)

        preds = skin_model.predict(img)
        idx = np.argmax(preds)
        confidence = float(np.max(preds))

        return jsonify({
            'class': skin_class_names[idx],
            'confidence': round(confidence * 100, 2)
        })

    elif model_type == "model2":  
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        features_values = []
        for feat in selected_features_breast:
            val = data.get(feat)
            if val is None:
                return jsonify({'error': f'Missing feature: {feat}'}), 400
            try:
                val = float(val)
            except ValueError:
                return jsonify({'error': f'Invalid value for feature: {feat}'}), 400
            features_values.append(val)

        arr = np.array(features_values).reshape(1, -1)
        pred = breast_model.predict(arr)[0]
        result = "Malignant" if pred == 1 else "Benign"

        return jsonify({
            'prediction': int(pred),
            'result': result
        })

    elif model_type == "model3":  # Lung Cancer 
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        features_values = []
        for feat in selected_features_lung:
            val = data.get(feat)
            if val is None:
                return jsonify({'error': f'Missing feature: {feat}'}), 400
            try:
                val = float(val)
            except ValueError:
                return jsonify({'error': f'Invalid value for feature: {feat}'}), 400
            features_values.append(val)

        arr = np.array(features_values).reshape(1, -1)
        pred = lung_cancer_model.predict(arr)[0]
        result = "Positive" if pred == 1 else "Negative"

        return jsonify({
            'prediction': int(pred),
            'result': result
        })

    else:
        return jsonify({'error': 'Unknown model type'}), 400


@app.route('/')
def home():
    return "🧪 LaboAI API Running!"

if __name__ == '__main__':
    app.run(debug=True)
