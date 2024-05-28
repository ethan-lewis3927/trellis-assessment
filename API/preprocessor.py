import re
import nltk
from nltk.corpus import stopwords

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in self.stop_words]
        return ' '.join(words)
