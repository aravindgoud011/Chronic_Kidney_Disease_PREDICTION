import numpy as np
import joblib
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from .preprocess import PREPROCESSED_PATH

def evaluate():
    data = np.load(PREPROCESSED_PATH)
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Load models
    encoder = keras.models.load_model("artifacts/models/encoder_model.keras")
    tabtransformer = keras.models.load_model("artifacts/models/tabtransformer_model.keras")
    meta_model = joblib.load("artifacts/models/meta_model_lr.pkl")
    meta_scaler = joblib.load("artifacts/scalers/meta_scaler.pkl")

    # Feature extraction
    ae_feats = encoder.predict(X_test)
    tt_feats = tabtransformer.predict(X_test)
    X_meta = np.concatenate([ae_feats, tt_feats], axis=1)

    # 🔑 IMPORTANT: scale meta features
    X_meta_scaled = meta_scaler.transform(X_meta)

    # Predictions
    y_pred = meta_model.predict(X_meta_scaled)
    y_probs = meta_model.predict_proba(X_meta_scaled)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, y_probs)

    print("\n✅ FINAL EVALUATION RESULTS ✅\n")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision : {precision_score(y_test, y_pred) * 100:.2f}%")
    print(f"Recall    : {recall_score(y_test, y_pred) * 100:.2f}%")
    print(f"F1-Score  : {f1_score(y_test, y_pred) * 100:.2f}%")
    print(f"AUC-ROC   : {auc * 100:.2f}%")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate()
