from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# === Load models ===
model_dir = "models"
scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))

rf_classifier = joblib.load(os.path.join(model_dir, "rf_classifier.joblib"))
rf_regressor = joblib.load(os.path.join(model_dir, "rf_regressor.joblib"))

nn_classifier = load_model(os.path.join(model_dir, "nn_classifier.h5"))
nn_regressor = load_model(os.path.join(model_dir, "nn_regressor.h5"))

# === Final features used ===
features = [
    'age',
    'gender',
    'annual_income',
    'occupation',
    'existing_bank_products',
    'kyc_completed',
    'relationship_years',
    'credit_score',
    'num_accounts',
    'avg_account_age',
    'dpd_30_count',
    'dpd_90_plus',
    'recent_default',
    'chronic_defaulter'
]

categorical_columns = ['gender', 'occupation']

@app.route('/')
def home():
    return "✅ Credit Card Approval API is running with 14 features."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Check for missing fields
        missing = [col for col in features if col not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Format input data
        X_input = [data[col] for col in features]
        X_df = np.array(X_input, dtype=object).reshape(1, -1)

        # Apply label encoding for categorical fields
        for col in categorical_columns:
            encoder = label_encoders[col]
            value = data[col]
            try:
                encoded_val = encoder.transform([value])[0]
                X_df[0, features.index(col)] = encoded_val
            except Exception:
                return jsonify({"error": f"Invalid value for '{col}': '{value}'"}), 400

        # Convert all fields to float
        X_df = X_df.astype(float)

        # Scale input
        X_scaled = scaler.transform(X_df)

        # Classification predictions
        nn_prob = nn_classifier.predict(X_scaled)[0][0]
        rf_prob = rf_classifier.predict_proba(X_df)[0][1]
        final_prob = (nn_prob + rf_prob) / 2
        approved = int(final_prob > 0.5)

        # If approved, predict credit limit
        if approved:
            nn_limit = nn_regressor.predict(X_scaled)[0][0]
            rf_limit = rf_regressor.predict(X_df)[0]
            final_limit = (nn_limit + rf_limit) / 2

            # Tiered suggestions
            safe = round(final_limit * 0.6, -2)
            optimal = round(final_limit * 0.8, -2)
            stretched = round(final_limit * 1.0, -2)
        else:
            final_limit = safe = optimal = stretched = 0

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
