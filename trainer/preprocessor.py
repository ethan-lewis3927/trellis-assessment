import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def read_data(self, directories):
        # Initialize an empty list to store the data
        data = []
        
        # Iterate over each directory
        for category in directories:
            directory_path = os.path.join(self.base_path, category)
            # Iterate over each file in the directory
            for filename in os.listdir(directory_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(directory_path, filename)
                    with open(file_path, 'r') as file:
                        text = file.read()
                        data.append({'text': text, 'category': category})
        
        # Convert the list to a pandas DataFrame
        df = pd.DataFrame(data, columns=['text', 'category'])
        return df

    def preprocess_text(self, text):
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Convert to lowercase
        text = text.lower()

        # Tokenize the text
        words = nltk.word_tokenize(text)

        # Remove stop words
        words = [word for word in words if word not in self.stop_words]

        # Join words back into a single string
        return ' '.join(words)

    def preprocess_dataframe(self, df):
        # Drop other rows
        rows_to_drop = df['category'].str.contains("other", case=False, na=False)
        df = df[~rows_to_drop]

        # Apply the preprocess_text function to the 'text' column
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        return df
    
