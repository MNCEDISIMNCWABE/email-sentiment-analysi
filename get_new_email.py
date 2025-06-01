import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
import pandas as pd
import logging
import os
from datetime import datetime, timedelta, timezone
import json

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

def load_last_processed_time():
    """Load the timestamp of the last processed email."""
    try:
        if os.path.exists('last_processed.json'):
            with open('last_processed.json', 'r') as f:
                data = json.load(f)
                # Parse as timezone-aware datetime
                dt = datetime.fromisoformat(data['last_processed'])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
        else:
            # If no previous run, start from 1 hour ago (timezone-aware)
            return datetime.now(timezone.utc) - timedelta(hours=1)
    except Exception as e:
        logging.warning(f"Could not load last processed time: {e}")
        return datetime.now(timezone.utc) - timedelta(hours=1)

def save_last_processed_time(timestamp):
    """Save the timestamp of the last processed email."""
    try:
        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        with open('last_processed.json', 'w') as f:
            json.dump({'last_processed': timestamp.isoformat()}, f)
    except Exception as e:
        logging.error(f"Could not save last processed time: {e}")

def normalize_datetime(dt):
    """Normalize datetime to timezone-aware UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        return dt.replace(tzinfo=timezone.utc)
    else:
        # Convert to UTC if it has timezone info
        return dt.astimezone(timezone.utc)

def get_new_emails_since(last_processed_time, imap_server='imap.gmail.com'):
    """Fetch emails received after the last processed time."""
    email_address, password = load_credentials()
    
    # Normalize last_processed_time to timezone-aware
    last_processed_time = normalize_datetime(last_processed_time)
    
    # Connect to the email server
    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(email_address, password)
    mail.select('inbox')
    
    try:
        # Search for emails since the last processed time
        # Use a broader date range to ensure we don't miss emails due to timezone issues
        since_date = (last_processed_time - timedelta(days=1)).strftime('%d-%b-%Y')
        status, email_ids = mail.search(None, f'SINCE {since_date}')
        
        if status != 'OK' or not email_ids[0]:
            logging.info("No new emails found")
            return pd.DataFrame()
        
        email_list = email_ids[0].split()
        new_emails = []
        latest_timestamp = last_processed_time
        
        for email_id in email_list:
            try:
                # Fetch email
                status, msg_data = mail.fetch(email_id, '(RFC822)')
                if status != 'OK':
                    continue
                    
                raw_email = msg_data[0][1]
                email_message = email.message_from_bytes(raw_email)
                
                # Extract date
                date_header = email_message.get("Date")
                if date_header:
                    email_date = parsedate_to_datetime(date_header)
                    email_date = normalize_datetime(email_date)
                    
                    # Only process emails newer than last processed time
                    if email_date <= last_processed_time:
                        continue
                    
                    # Update latest timestamp
                    if email_date > latest_timestamp:
                        latest_timestamp = email_date
                else:
                    continue
                
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
                
                email_data = {
                    'from_email': from_email,
                    'subject': subject,
                    'body': body,
                    'date_sent': email_date,
                    'email_id': email_id.decode()
                }
                
                new_emails.append(email_data)
                
            except Exception as e:
                logging.error(f"Error processing email {email_id}: {e}")
                continue
        
        # Save the latest timestamp
        if latest_timestamp > last_processed_time:
            save_last_processed_time(latest_timestamp)
        
        if new_emails:
            logging.info(f"Found {len(new_emails)} new emails")
            return pd.DataFrame(new_emails)
        else:
            logging.info("No new emails to process")
            return pd.DataFrame()
            
    finally:
        mail.close()
        mail.logout()

def get_latest_email(imap_server='imap.gmail.com'):
    """Wrapper function to maintain compatibility - gets new emails since last run."""
    last_processed = load_last_processed_time()
    return get_new_emails_since(last_processed, imap_server)
