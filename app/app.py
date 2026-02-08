from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCALER_PATH = os.path.join(BASE_DIR, "../artifacts/scalers/scaler.pkl")
META_SCALER_PATH = os.path.join(BASE_DIR, "../artifacts/scalers/meta_scaler.pkl")

ENCODER_PATH = os.path.join(BASE_DIR, "../artifacts/models/encoder_model.keras")
TABTRANSFORMER_PATH = os.path.join(BASE_DIR, "../artifacts/models/tabtransformer_model.keras")
META_MODEL_PATH = os.path.join(BASE_DIR, "../artifacts/models/meta_model_lr.pkl")

scaler = joblib.load(SCALER_PATH)
meta_scaler = joblib.load(META_SCALER_PATH)

encoder = load_model(ENCODER_PATH)
tabtransformer = load_model(TABTRANSFORMER_PATH)
meta_model = joblib.load(META_MODEL_PATH)

THRESHOLD = 0.55   # stable ~85%

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = np.array([[
            float(request.form['age']),
            float(request.form['bp']),
            float(request.form['sg']),
            float(request.form['al']),
            float(request.form['su']),
            float(request.form['bgr']),
            float(request.form['bu']),
            float(request.form['sc']),
            float(request.form['sod']),
            float(request.form['pot']),
            float(request.form['hemo']),
            float(request.form['pcv']),
            float(request.form['wbcc']),
            float(request.form['rbcc']),
            float(request.form['hypertension'])
        ]])

        # 🩺 MEDICAL RULE SAFETY NET (PRIMARY)
        if (
            features[0][7] <= 1.2 and      # creatinine
            features[0][6] <= 40 and       # urea
            features[0][10] >= 13 and      # hemoglobin
            features[0][2] >= 1.020 and    # sg
            features[0][3] == 0 and        # albumin
            features[0][14] == 0           # hypertension
        ):
            return render_template(
                'result.html',
                result="✅ No CKD Detected (Medical rule validation)"
            )

        # 🔍 ML PIPELINE (SECONDARY)
        scaled = scaler.transform(features)

        ae_feat = encoder.predict(scaled)
        tt_feat = tabtransformer.predict(scaled)

        meta_feat = np.concatenate([ae_feat, tt_feat], axis=1)
        meta_feat = meta_scaler.transform(meta_feat)

        prob = meta_model.predict_proba(meta_feat)[0][1]
        confidence = round(prob * 100, 2)

        if prob >= THRESHOLD:
            result = f"⚠️ CKD Detected (Confidence: {confidence}%)"
        else:
            result = f"✅ No CKD Detected (Confidence: {confidence}%)"

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('result.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
