from flask import Blueprint, request, jsonify
import joblib
import numpy as np
from preprocessor import Preprocessor

predict_blueprint = Blueprint('predict', __name__)

# Load the trained model, vectorizer, and label encoder
model = joblib.load('../trainer/models/log_reg_model.pkl')
vectorizer = joblib.load('../trainer/models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('../trainer/models/label_encoder.pkl')
num_labels = len(label_encoder.classes_)

# Create Preprocessor instance
preprocessor = Preprocessor()

@predict_blueprint.route('/classify_document', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input
        if 'document_text' not in data or data['document_text'] == '':
            return jsonify({"error": "Missing 'document_text' in request body"}), 400
        
        text = data['document_text']

        # Preprocess the text
        preprocessed_text = preprocessor.preprocess_text(text)

        # Vectorize the preprocessed text
        X = vectorizer.transform([preprocessed_text])

        # Predict the label
        pred_probabilities = model.predict_proba(X)
        preds = np.argmax(pred_probabilities, axis=1)

        # Check confidence for 'other' label
        confidence_threshold = 0.35
        max_proba = np.max(pred_probabilities, axis=1)[0]
        if max_proba < confidence_threshold:
            predicted_label = 'other'
        else:
            predicted_label = label_encoder.inverse_transform(preds)[0]

        response = {
            'message': 'success',
            'predicted_label': predicted_label,
            'confidence': max_proba
        }

        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500