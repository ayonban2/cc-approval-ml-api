
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging

app = Flask(__name__)

# More explicit CORS configuration
CORS(app, 
     origins="*",  # Allow all origins for testing
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'Accept', 'Origin', 'X-Requested-With'],
     supports_credentials=False)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load models ===
model_dir = "models"
try:
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))
    
    rf_classifier = joblib.load(os.path.join(model_dir, "rf_classifier.joblib"))
    rf_regressor = joblib.load(os.path.join(model_dir, "rf_regressor.joblib"))
    
    nn_classifier = load_model(os.path.join(model_dir, "nn_classifier.h5"))
    nn_regressor = load_model(os.path.join(model_dir, "nn_regressor.h5"))
    
    logger.info("✅ All models loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading models: {e}")
    raise

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
    logger.info("Root endpoint hit")
    return jsonify({
        "status": "✅ Credit Card Approval API is running with 14 features",
        "features": features,
        "version": "1.0.0"
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "features_count": len(features)})

# Explicit OPTIONS handler for all routes
@app.route('/predict', methods=['OPTIONS'])
def handle_options():
    logger.info("OPTIONS request received for /predict")
    response = jsonify({"status": "ok"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept, Origin, X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response, 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info(f"POST request received: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request origin: {request.headers.get('Origin', 'No origin header')}")
        
        data = request.json
        logger.info(f"Request data: {data}")

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Check for missing fields
        missing = [col for col in features if col not in data]
        if missing:
            logger.error(f"Missing fields: {missing}")
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
                logger.info(f"Encoded {col}: {value} -> {encoded_val}")
            except Exception as e:
                logger.error(f"Encoding error for {col}: {e}")
                return jsonify({"error": f"Invalid value for '{col}': '{value}'"}), 400

        # Convert all fields to float
        X_df = X_df.astype(float)
        logger.info(f"Processed input shape: {X_df.shape}")

        # Scale input
        X_scaled = scaler.transform(X_df)

        # Classification predictions
        nn_prob = nn_classifier.predict(X_scaled)[0][0]
        rf_prob = rf_classifier.predict_proba(X_df)[0][1]
        final_prob = (nn_prob + rf_prob) / 2
        approved = int(final_prob > 0.5)

        logger.info(f"NN probability: {nn_prob}, RF probability: {rf_prob}, Final: {final_prob}")

        # If approved, predict credit limit
        if approved:
            nn_limit = nn_regressor.predict(X_scaled)[0][0]
            rf_limit = rf_regressor.predict(X_df)[0]
            final_limit = (nn_limit + rf_limit) / 2

            # Tiered suggestions
            safe = round(final_limit * 0.6, -2)
            optimal = round(final_limit * 0.8, -2)
            stretched = round(final_limit * 1.0, -2)
            
            logger.info(f"Credit limits - NN: {nn_limit}, RF: {rf_limit}, Final: {final_limit}")
        else:
            final_limit = safe = optimal = stretched = 0

        result = {
            "approved": bool(approved),
            "approval_probability": round(final_prob, 3),
            "credit_limit": int(round(final_limit)),
            "limit_suggestions": {
                "safe": int(safe),
                "optimal": int(optimal),
                "stretched": int(stretched)
            }
        }
        
        logger.info(f"Returning result: {result}")
        
        # Create response with explicit CORS headers
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        error_response = jsonify({"error": str(e)})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept, Origin, X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

@app.errorhandler(404)
def not_found(error):
    response = jsonify({"error": "Endpoint not found"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 404

@app.errorhandler(500)
def internal_error(error):
    response = jsonify({"error": "Internal server error"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
