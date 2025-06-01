import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
import pandas as pd
import yaml
import logging
import os

def load_credentials():
    """Load email credentials from environment variables."""
    try:
        email_address = os.getenv('EMAIL_ADDRESS')
        password = os.getenv('EMAIL_PASSWORD')
        
        if not email_address or not password:
            raise ValueError("EMAIL_ADDRESS and EMAIL_PASSWORD environment variables must be set")
        
        return email_address, password
    except Exception as e:
        logging.error(f"Failed to load credentials: {e}")
        raise

def get_latest_email(imap_server='imap.gmail.com'):
    """Fetch the latest email from the inbox and return it as a DataFrame."""
    # Load credentials
    email_address, password = load_credentials()

    # Connect to the email server
    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(email_address, password)
    mail.select('inbox')

    try:
        # Search for all emails and get the latest one
        status, email_ids = mail.search(None, 'ALL')
        if status != 'OK':
            raise Exception("Failed to search emails")

        latest_email_id = email_ids[0].split()[-1]  # Get the most recent email ID

        # Fetch the latest email
        status, msg_data = mail.fetch(latest_email_id, '(RFC822)')
        if status != 'OK':
            raise Exception("Failed to fetch email")

        raw_email = msg_data[0][1]
        email_message = email.message_from_bytes(raw_email)

        # Extract email details
        from_header = email_message.get("From", "")
        from_parts = decode_header(from_header)
        from_email = ''.join(
            part.decode(encoding or 'utf-8', errors='ignore') if isinstance(part, bytes) else part
            for part, encoding in from_parts
        )

        subject_header = email_message.get("Subject", "")
        subject_parts = decode_header(subject_header)
        subject = ''.join(
            part.decode(encoding or 'utf-8', errors='ignore') if isinstance(part, bytes) else part
            for part, encoding in subject_parts
        )

        # Extract email body
        body = ""
        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode(errors='ignore')
                        break
                    except:
                        continue
        else:
            try:
                body = email_message.get_payload(decode=True).decode(errors='ignore')
            except:
                pass

        # Extract date
        date_sent = email_message.get("Date")
        date_sent = parsedate_to_datetime(date_sent) if date_sent else None

        # Create a DataFrame with the email details
        email_data = {
            'from_email': from_email,
            'subject': subject,
            'body': body,
            'date_sent': date_sent
        }

        df = pd.DataFrame([email_data])
        return df

    finally:
        mail.close()
        mail.logout()