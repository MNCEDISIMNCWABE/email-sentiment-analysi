import logging
import pandas as pd
from get_new_email import get_latest_email
from model_training import clean_text, text_preprocessing, load_model, predict_email_sentiment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_sentiment_for_new_emails(model_path, new_emails_df):
    """
    Predict sentiment for new emails using a trained model.
    :param model_path: Path to the trained model file.
    :param new_emails_df: A DataFrame containing 'subject' and 'body' columns of new emails.
    :return: A list of sentiment predictions for the new emails.
    """
    try:
        # Load the trained model
        model = load_model(model_path)
        predictions = []
        
        for idx, email in new_emails_df.iterrows():  # Fixed syntax error here
            subject = email.get('subject', '')
            body = email.get('body', '')
            # Predict sentiment
            prediction = predict_email_sentiment(model, body, subject)
            predictions.append(prediction)
        return predictions
    except Exception as e:
        logger.error(f"Error predicting sentiment for new emails: {e}", exc_info=True)
        return []

def main():
    logger.info("Starting email sentiment analysis...")
    
    model_path = 'email_sentiment_model.pkl'
    
    try:
        new_emails_df = get_latest_email()
        
        # Check if there are any new emails
        if new_emails_df.empty:
            logger.info("No new emails to process.")
            return
        
        logger.info(f"Processing {len(new_emails_df)} new email(s)")
        
        predictions = predict_sentiment_for_new_emails(model_path, new_emails_df)
        
        if not predictions:
            logger.warning("No predictions generated")
            return
        
        # Process each email and its prediction
        for idx, prediction in enumerate(predictions):
            email_details = new_emails_df.iloc[idx]
            date_sent = email_details['date_sent']
            from_email = email_details['from_email']
            subject = email_details['subject']
            
            logger.info(f"=== EMAIL {idx + 1} ===")
            logger.info("Date Sent: %s", date_sent)
            logger.info("From: %s", from_email)
            logger.info("Subject: %s", subject)
            logger.info("Sentiment: %s", prediction['sentiment'])
            logger.info("Confidence: %.2f%%", prediction['confidence'] * 100)
            logger.info("=" * 50)
        
        logger.info("Email sentiment analysis completed.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()
