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
        for _, email in new_emails_df.iterrows():
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

    model_path = 'email_sentiment_model.pkl'
    new_email_df = get_latest_email('credentials.yaml')

    predictions = predict_sentiment_for_new_emails(model_path, new_email_df)
    for idx, prediction in enumerate(predictions):
        email_details = new_email_df.iloc[idx]
        date_sent = email_details['date_sent']
        from_email = email_details['from_email']
        subject = email_details['subject']

        logger.info("Date Sent: %s", date_sent)
        logger.info("From: %s", from_email)
        logger.info("Subject: %s", subject)
        logger.info("Sentiment: %s", prediction['sentiment'])
        logger.info("Confidence: %.2f%%\n", prediction['confidence'] * 100)

if __name__ == "__main__":
    main()