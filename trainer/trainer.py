import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessor import Preprocessor
import joblib

class Trainer:
    def __init__(self, df):
        self.df = df
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = None
        self.calibrated_model = None
        self.num_labels = None

    def preprocess_labels(self):
        self.df['label'] = self.label_encoder.fit_transform(self.df['category'])
        self.num_labels = len(self.label_encoder.classes_)

    def vectorize_text(self):
        self.X = self.vectorizer.fit_transform(self.df['text'])
        self.y = self.df['label']

    def split_data(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.model.fit(self.X_train, self.y_train)
        self.calibrated_model = CalibratedClassifierCV(self.model, cv='prefit')
        self.calibrated_model.fit(self.X_train, self.y_train)

    def evaluate_model(self, confidence_threshold=0.35):
        y_proba = self.calibrated_model.predict_proba(self.X_val)
        y_pred_log_reg = np.argmax(y_proba, axis=1)

        y_pred_with_other = []
        for i, proba in enumerate(y_proba):
            max_proba = np.max(proba)
            if max_proba < confidence_threshold:
                y_pred_with_other.append(self.num_labels)  # Assign 'other' class
            else:
                y_pred_with_other.append(y_pred_log_reg[i])

        class_names = list(self.label_encoder.classes_) + ['other']
        print("Log Reg Model with 'other' class:")
        print(classification_report(self.y_val, y_pred_with_other, target_names=class_names, zero_division=0.0))

    def save_model(self, model_path, vectorizer_path, label_encoder_path, model_save_path):
        joblib.dump(self.calibrated_model, model_save_path + model_path)
        joblib.dump(self.vectorizer, model_save_path + vectorizer_path)
        joblib.dump(self.label_encoder, model_save_path + label_encoder_path)

# Example usage
if __name__ == "__main__":
    base_path = '../trellis_assessment_ds'
    model_save_path = 'models/'
    directories = ['business', 'entertainment', 'food', 'graphics', 'historical', 'medical', 'other', 'politics', 'space', 'sport', 'technologie']

    # Define the preprocessor
    preprocessor = Preprocessor(base_path)
    
    # Read data
    df = preprocessor.read_data(directories)
    
    # Preprocess text
    df = preprocessor.preprocess_dataframe(df)
    
    print("Data preprocessing complete...")
    
    # Define the trainer
    trainer = Trainer(df)

    # Encode categories as lables
    trainer.preprocess_labels()

    # Fit the TD-IDF vectorizer and save the vectorizer as a pkl file for API use
    trainer.vectorize_text()

    # Split data into training and validation sets
    trainer.split_data()

    # Train the model 
    trainer.train_model()

    # Run eval 
    trainer.evaluate_model()

    # Save the model and vectorizer
    trainer.save_model('log_reg_model.pkl', 'tfidf_vectorizer.pkl', 'label_encoder.pkl', model_save_path)

