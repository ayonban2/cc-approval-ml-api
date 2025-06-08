from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# === Load models and objects ===
model_dir = "models"

scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))

rf_classifier = joblib.load(os.path.join(model_dir, "rf_classifier.joblib"))
rf_regressor = joblib.load(os.path.join(model_dir, "rf_regressor.joblib"))

nn_classifier = load_model(os.path.join(model_dir, "nn_classifier.h5"))
nn_regressor = load_model(os.path.join(model_dir, "nn_regressor.h5"))

# === Feature order ===
feature_columns = ['age', 'gender', 'income', 'existing_loans', 'credit_score']

@app.route('/')
def home():
    return "Credit Card Approval API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Input validation
        missing = [col for col in feature_columns if col not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Extract features in correct order
        X_input = [data[col] for col in feature_columns]
        X_df = np.array(X_input).reshape(1, -1)

        # Apply label encoding
        for i, col in enumerate(['gender', 'occupation']):
            encoder = label_encoders[col]
            X_df[0, feature_columns.index(col)] = encoder.transform([data[col]])[0]

        # Convert to float
        X_df = X_df.astype(float)

        # Scale for NN
        X_scaled = scaler.transform(X_df)

        # Predict approval
        nn_prob = nn_classifier.predict(X_scaled)[0][0]
        rf_prob = rf_classifier.predict_proba(X_df)[0][1]
        final_prob = (nn_prob + rf_prob) / 2
        approved = int(final_prob > 0.5)

        if approved:
            # Predict credit limit
            nn_limit = nn_regressor.predict(X_scaled)[0][0]
            rf_limit = rf_regressor.predict(X_df)[0]
            final_limit = (nn_limit + rf_limit) / 2

            # Suggest tiers
            safe = round(final_limit * 0.6, -2)
            optimal = round(final_limit * 0.8, -2)
            stretched = round(final_limit * 1.0, -2)
        else:
            final_limit = 0
            safe = optimal = stretched = 0

        return jsonify({
            "approved": bool(approved),
            "approval_probability": round(final_prob, 3),
            "credit_limit": round(final_limit),
            "limit_suggestions": {
                "safe": safe,
                "optimal": optimal,
                "stretched": stretched
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
