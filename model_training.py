import logging
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('email_sentiment_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    logger.info("Downloaded NLTK punkt tokenizer")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    logger.info("Downloaded NLTK stopwords")

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    logger.info("Downloaded NLTK wordnet")

def clean_text(text):
    """Clean and preprocess email text"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters but keep emojis (basic approach)
    text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def text_preprocessing(text):
    """Advanced text preprocessing with tokenization, stopword removal, and lemmatization"""
    if not text:
        return ""
    
    try:
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in text preprocessing: {e}", exc_info=True)
        return ""

def create_email_features(df):
    """Create combined email features from subject and body"""
    df = df.copy()

    try:
        # Fill missing values
        df['subject'] = df['subject'].fillna('')
        df['body'] = df['body'].fillna('')

        # Clean text
        df['subject_clean'] = df['subject'].apply(clean_text)
        df['body_clean'] = df['body'].apply(clean_text)

        # Combine subject and body 
        df['combined_text'] = df['subject_clean'] + ' ' + df['subject_clean'] + ' ' + df['body_clean']

        # Advanced preprocessing
        df['processed_text'] = df['combined_text'].apply(text_preprocessing)

        return df
    except Exception as e:
        logger.error(f"Error creating email features: {e}", exc_info=True)
        raise

def get_textblob_sentiment(text):
    """Get sentiment using TextBlob as baseline"""
    if not text:
        return 'neutral'
    
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    except Exception as e:
        logger.warning(f"TextBlob sentiment analysis failed: {e}")
        return 'neutral'

def create_sentiment_labels(df, method='textblob'):
    """Create sentiment labels for training data"""
    df = df.copy()
    
    try:
        if method == 'textblob':
            logger.info("Creating sentiment labels using TextBlob")
            df['sentiment'] = df['processed_text'].apply(get_textblob_sentiment)
            logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
        
        return df
    except Exception as e:
        logger.error(f"Error creating sentiment labels: {e}", exc_info=True)
        raise

def build_sentiment_model(X_train, y_train, model_type='logistic'):
    """Build and train sentiment analysis model"""
    try:
        logger.info(f"Building {model_type} sentiment model")
        
        if model_type == 'logistic':
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ])
        elif model_type == 'random_forest':
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
        elif model_type == 'svm':
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
                ('classifier', SVC(kernel='linear', random_state=42, probability=True))
            ])
        
        model.fit(X_train, y_train)
        logger.info(f"Successfully trained {model_type} model")
        return model
    except Exception as e:
        logger.error(f"Error building model: {e}", exc_info=True)
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    try:
        logger.info("Evaluating model performance")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n" + report)
        logger.debug("Confusion Matrix:\n" + str(confusion_mat))
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion_mat,
            'predictions': y_pred
        }
    except Exception as e:
        logger.error(f"Error evaluating model: {e}", exc_info=True)
        raise

def predict_email_sentiment(model, email_text, subject_text=""):
    """Predict sentiment for new email"""
    try:
        # Combine subject and body
        combined = clean_text(subject_text) + ' ' + clean_text(subject_text) + ' ' + clean_text(email_text)
        processed = text_preprocessing(combined)
        
        # Predict
        prediction = model.predict([processed])[0]
        probability = model.predict_proba([processed])[0]
        
        logger.debug(f"Prediction: {prediction}, Probabilities: {dict(zip(model.classes_, probability))}")
        
        return {
            'sentiment': prediction,
            'confidence': max(probability),
            'probabilities': dict(zip(model.classes_, probability))
        }
    except Exception as e:
        logger.error(f"Error predicting sentiment: {e}", exc_info=True)
        return {
            'sentiment': 'error',
            'confidence': 0,
            'probabilities': {}
        }

def save_model(model, filepath):
    """Save trained model to file"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {e}", exc_info=True)
        raise

def load_model(filepath):
    """Load trained model from file"""
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise

def full_pipeline(df_emails, test_size=0.2, model_type='logistic', save_path=None):
    """Complete pipeline for email sentiment analysis"""
    logger.info("Starting email sentiment analysis pipeline")
    
    try:
        # Step 1: Feature engineering
        logger.info("Creating email features")
        df_processed = create_email_features(df_emails)
        
        # Step 2: Create sentiment labels
        logger.info("Creating sentiment labels")
        df_labeled = create_sentiment_labels(df_processed)
        
        # Step 3: Prepare data for training
        logger.info("Preparing training data")
        X = df_labeled['processed_text']
        y = df_labeled['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(f"Sentiment distribution: {y.value_counts().to_dict()}")
        
        # Step 4: Build and train model
        logger.info(f"Training {model_type} model")
        model = build_sentiment_model(X_train, y_train, model_type)
        
        # Step 5: Evaluate model
        logger.info("Evaluating model")
        evaluation = evaluate_model(model, X_test, y_test)
        
        # Step 6: Save model if path provided
        if save_path:
            logger.info(f"Saving model to {save_path}")
            save_model(model, save_path)
        
        return {
            'model': model,
            'evaluation': evaluation,
            'processed_data': df_labeled
        }
    except Exception as e:
        logger.error("Pipeline failed with error", exc_info=True)
        raise