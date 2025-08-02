#!/usr/bin/env python3
"""
Enhanced Email Assistant - Core Module
AI-powered email classification, response generation, and automation
"""

import os
import json
import pickle
import base64
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Google API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# OpenAI for AI responses
import openai
from openai import OpenAI

# Other imports
import logging
from pathlib import Path

# Email classification categories
EMAIL_CATEGORIES = {
    'URGENT': {'priority': 1, 'color': '#FF4444', 'icon': '🚨'},
    'IMPORTANT': {'priority': 2, 'color': '#FF8800', 'icon': '⭐'},
    'FOLLOW_UP': {'priority': 3, 'color': '#4488FF', 'icon': '🔄'},
    'CUSTOMER_SUPPORT': {'priority': 4, 'color': '#00AA44', 'icon': '🎧'},
    'MARKETING': {'priority': 5, 'color': '#8844FF', 'icon': '📢'},
    'SPAM': {'priority': 6, 'color': '#888888', 'icon': '🗑️'},
    'NEWSLETTER': {'priority': 7, 'color': '#44AAFF', 'icon': '📰'},
    'SOCIAL': {'priority': 8, 'color': '#FF44AA', 'icon': '👥'},
    'PERSONAL': {'priority': 9, 'color': '#00DDAA', 'icon': '💌'},
    'OTHER': {'priority': 10, 'color': '#AAAAAA', 'icon': '📄'}
}

@dataclass
class EmailData:
    """Email data structure"""
    id: str
    subject: str
    sender: str
    body: str
    timestamp: datetime
    thread_id: str
    labels: List[str]
    snippet: str
    attachments: List[str]
    is_unread: bool

@dataclass
class EmailClassification:
    """Email classification result"""
    category: str
    confidence: float
    urgency_score: int  # 1-10 scale
    requires_response: bool
    sentiment: str  # positive, neutral, negative
    keywords: List[str]
    suggested_actions: List[str]

@dataclass
class EmailResponse:
    """Generated email response"""
    subject: str
    body: str
    tone: str
    confidence: float
    auto_send_recommended: bool
    reasoning: str

class EnhancedEmailAssistant:
    """Enhanced AI Email Assistant with classification and response generation"""
    
    def __init__(self, config_file: str = "email_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.setup_logging()
        self.gmail_service = None
        self.openai_client = None
        self.stats = {
            'emails_processed': 0,
            'responses_generated': 0,
            'auto_sent': 0,
            'categories': {},
            'last_run': None
        }
        
        # Initialize services
        self.initialize_gmail()
        self.initialize_openai()
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            "gmail": {
                "scopes": ["https://www.googleapis.com/auth/gmail.readonly",
                          "https://www.googleapis.com/auth/gmail.send",
                          "https://www.googleapis.com/auth/gmail.modify"],
                "credentials_file": "credentials.json",
                "token_file": "token.pickle"
            },
            "openai": {
                "api_key": "",
                "model": "gpt-4o-mini",
                "max_tokens": 500,
                "temperature": 0.7
            },
            "processing": {
                "max_emails_per_run": 10,
                "hours_lookback": 24,
                "auto_send_threshold": 0.85,
                "response_style": "professional"
            },
            "filters": {
                "priority_senders": [],
                "excluded_senders": ["noreply@", "no-reply@", "donotreply@"],
                "spam_keywords": ["lottery", "winner", "congratulations", "urgent transfer"],
                "urgent_keywords": ["urgent", "asap", "emergency", "immediate"]
            },
            "responses": {
                "styles": {
                    "professional": "Professional and formal tone",
                    "casual": "Friendly and conversational tone", 
                    "concise": "Brief and to-the-point responses"
                },
                "templates": {
                    "acknowledgment": "Thank you for your email. I have received your message and will respond shortly.",
                    "meeting_request": "Thank you for the meeting request. I'll check my calendar and get back to you.",
                    "follow_up": "Following up on our previous conversation..."
                }
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    self._deep_update(default_config, loaded_config)
            except Exception as e:
                logging.error(f"Error loading config: {e}")
        
        # Save updated config
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config: Dict = None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('email_assistant.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_gmail(self):
        """Initialize Gmail API service"""
        try:
            creds = None
            token_file = self.config['gmail']['token_file']
            credentials_file = self.config['gmail']['credentials_file']
            scopes = self.config['gmail']['scopes']
            
            # Load existing token
            if os.path.exists(token_file):
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)
            
            # If no valid credentials, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(credentials_file):
                        self.logger.error(f"Gmail credentials file not found: {credentials_file}")
                        self.logger.info("Please download credentials.json from Google Cloud Console")
                        return False
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        credentials_file, scopes)
                    creds = flow.run_local_server(port=0)
                
                # Save credentials
                with open(token_file, 'wb') as token:
                    pickle.dump(creds, token)
            
            self.gmail_service = build('gmail', 'v1', credentials=creds)
            self.logger.info("Gmail API initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gmail API: {e}")
            return False
    
    def initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            api_key = self.config['openai']['api_key']
            if not api_key:
                self.logger.error("OpenAI API key not found in config")
                self.logger.info("Please add your OpenAI API key to the config file")
                return False
            
            self.openai_client = OpenAI(api_key=api_key)
            self.logger.info("OpenAI client initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI: {e}")
            return False
    
    def get_recent_emails(self, max_results: int = None) -> List[EmailData]:
        """Fetch recent emails from Gmail"""
        if not self.gmail_service:
            self.logger.error("Gmail service not initialized")
            return []
        
        if max_results is None:
            max_results = self.config['processing']['max_emails_per_run']
        
        try:
            # Calculate time range
            hours_back = self.config['processing']['hours_lookback']
            after_date = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y/%m/%d')
            
            # Search for unread emails
            query = f'is:unread after:{after_date}'
            
            # Get message list
            results = self.gmail_service.users().messages().list(
                userId='me', q=query, maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for message in messages:
                email_data = self._fetch_email_details(message['id'])
                if email_data and self._should_process_email(email_data):
                    emails.append(email_data)
            
            self.logger.info(f"Fetched {len(emails)} emails for processing")
            return emails
            
        except HttpError as e:
            self.logger.error(f"Gmail API error: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching emails: {e}")
            return []
    
    def _fetch_email_details(self, message_id: str) -> Optional[EmailData]:
        """Fetch detailed email information"""
        try:
            message = self.gmail_service.users().messages().get(
                userId='me', id=message_id, format='full'
            ).execute()
            
            headers = message['payload'].get('headers', [])
            
            # Extract headers
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
            date_str = next((h['value'] for h in headers if h['name'] == 'Date'), '')
            
            # Parse date
            try:
                timestamp = datetime.strptime(date_str[:25], '%a, %d %b %Y %H:%M:%S')
            except:
                timestamp = datetime.now()
            
            # Extract body
            body = self._extract_email_body(message['payload'])
            
            # Extract other info
            snippet = message.get('snippet', '')
            thread_id = message.get('threadId', '')
            labels = message.get('labelIds', [])
            is_unread = 'UNREAD' in labels
            
            return EmailData(
                id=message_id,
                subject=subject,
                sender=sender,
                body=body,
                timestamp=timestamp,
                thread_id=thread_id,
                labels=labels,
                snippet=snippet,
                attachments=[],  # TODO: Extract attachments
                is_unread=is_unread
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching email {message_id}: {e}")
            return None
    
    def _extract_email_body(self, payload: Dict) -> str:
        """Extract email body from payload"""
        body = ""
        
        if 'body' in payload and 'data' in payload['body']:
            # Single part message
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        elif 'parts' in payload:
            # Multi-part message
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                elif part['mimeType'] == 'text/html':
                    if 'data' in part['body'] and not body:  # Use HTML only if no plain text
                        html_content = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        # Simple HTML to text conversion
                        body = re.sub('<[^<]+?>', '', html_content)
        
        return body.strip()
    
    def _should_process_email(self, email: EmailData) -> bool:
        """Determine if email should be processed"""
        # Check excluded senders
        excluded = self.config['filters']['excluded_senders']
        for excluded_sender in excluded:
            if excluded_sender.lower() in email.sender.lower():
                return False
        
        # Check if it's too old
        hours_back = self.config['processing']['hours_lookback']
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        if email.timestamp < cutoff_time:
            return False
        
        # Check if it has minimal content
        if len(email.body.strip()) < 10 and len(email.subject.strip()) < 5:
            return False
        
        return True
    
    def classify_email(self, email: EmailData) -> EmailClassification:
        """Classify email using AI"""
        if not self.openai_client:
            return self._fallback_classification(email)
        
        try:
            # Prepare classification prompt
            prompt = self._build_classification_prompt(email)
            
            response = self.openai_client.chat.completions.create(
                model=self.config['openai']['model'],
                messages=[
                    {"role": "system", "content": "You are an expert email classifier. Analyze emails and provide structured classification."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            # Parse AI response
            classification_text = response.choices[0].message.content
            return self._parse_classification_response(classification_text, email)
            
        except Exception as e:
            self.logger.error(f"Error in AI classification: {e}")
            return self._fallback_classification(email)
    
    def _build_classification_prompt(self, email: EmailData) -> str:
        """Build prompt for email classification"""
        return f"""
Classify this email and provide the analysis in the following format:

CATEGORY: [Choose from: URGENT, IMPORTANT, FOLLOW_UP, CUSTOMER_SUPPORT, MARKETING, SPAM, NEWSLETTER, SOCIAL, PERSONAL, OTHER]
CONFIDENCE: [0.0-1.0]
URGENCY: [1-10 scale]
REQUIRES_RESPONSE: [true/false]
SENTIMENT: [positive/neutral/negative]
KEYWORDS: [comma-separated relevant keywords]
ACTIONS: [comma-separated suggested actions]

Email Details:
Subject: {email.subject}
Sender: {email.sender}
Content: {email.body[:500]}...

Consider:
- Urgency indicators (urgent, asap, emergency)
- Sender reputation and domain
- Content type (personal, business, automated)
- Response requirements
- Spam indicators
"""
    
    def _parse_classification_response(self, response_text: str, email: EmailData) -> EmailClassification:
        """Parse AI classification response"""
        try:
            lines = response_text.strip().split('\n')
            result = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    result[key.strip().lower()] = value.strip()
            
            category = result.get('category', 'OTHER').upper()
            if category not in EMAIL_CATEGORIES:
                category = 'OTHER'
            
            confidence = float(result.get('confidence', '0.5'))
            urgency = int(result.get('urgency', '5'))
            requires_response = result.get('requires_response', 'false').lower() == 'true'
            sentiment = result.get('sentiment', 'neutral').lower()
            
            keywords = []
            if 'keywords' in result:
                keywords = [k.strip() for k in result['keywords'].split(',')]
            
            actions = []
            if 'actions' in result:
                actions = [a.strip() for a in result['actions'].split(',')]
            
            return EmailClassification(
                category=category,
                confidence=confidence,
                urgency_score=urgency,
                requires_response=requires_response,
                sentiment=sentiment,
                keywords=keywords,
                suggested_actions=actions
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing classification: {e}")
            return self._fallback_classification(email)
    
    def _fallback_classification(self, email: EmailData) -> EmailClassification:
        """Fallback classification using rules"""
        category = 'OTHER'
        urgency = 5
        requires_response = True
        
        # Simple rule-based classification
        subject_lower = email.subject.lower()
        body_lower = email.body.lower()
        sender_lower = email.sender.lower()
        
        # Check for urgent keywords
        urgent_keywords = self.config['filters']['urgent_keywords']
        if any(keyword in subject_lower or keyword in body_lower for keyword in urgent_keywords):
            category = 'URGENT'
            urgency = 9
        
        # Check for spam indicators
        spam_keywords = self.config['filters']['spam_keywords']
        if any(keyword in subject_lower or keyword in body_lower for keyword in spam_keywords):
            category = 'SPAM'
            urgency = 1
            requires_response = False
        
        # Check for newsletters/marketing
        if any(word in sender_lower for word in ['newsletter', 'marketing', 'promo']):
            category = 'NEWSLETTER'
            urgency = 2
            requires_response = False
        
        # Check for automated emails
        if any(word in sender_lower for word in ['noreply', 'no-reply', 'donotreply']):
            category = 'OTHER'
            requires_response = False
        
        return EmailClassification(
            category=category,
            confidence=0.6,
            urgency_score=urgency,
            requires_response=requires_response,
            sentiment='neutral',
            keywords=[],
            suggested_actions=['Review and respond' if requires_response else 'Archive']
        )
    
    def generate_response(self, email: EmailData, classification: EmailClassification) -> Optional[EmailResponse]:
        """Generate AI response for email"""
        if not classification.requires_response or not self.openai_client:
            return None
        
        try:
            # Build response prompt
            prompt = self._build_response_prompt(email, classification)
            
            response = self.openai_client.chat.completions.create(
                model=self.config['openai']['model'],
                messages=[
                    {"role": "system", "content": "You are a professional email assistant. Generate appropriate email responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config['openai']['max_tokens'],
                temperature=self.config['openai']['temperature']
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            return self._parse_response_generation(response_text, email, classification)
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return None
    
    def _build_response_prompt(self, email: EmailData, classification: EmailClassification) -> str:
        """Build prompt for response generation"""
        style = self.config['processing']['response_style']
        style_description = self.config['responses']['styles'].get(style, 'Professional tone')
        
        return f"""
Generate an appropriate email response using the following format:

SUBJECT: [Response subject line]
BODY: [Response body]
TONE: [professional/casual/formal]
CONFIDENCE: [0.0-1.0 - how confident you are this response is appropriate]
AUTO_SEND: [true/false - whether this response is safe to send automatically]
REASONING: [Brief explanation of the response approach]

Guidelines:
- Style: {style_description}
- Category: {classification.category}
- Urgency: {classification.urgency_score}/10
- Sentiment: {classification.sentiment}

Original Email:
Subject: {email.subject}
From: {email.sender}
Content: {email.body[:800]}

Response Requirements:
- Be helpful and professional
- Address the main points
- Keep it concise but complete
- Match the appropriate tone
- Include relevant next steps if needed
"""
    
    def _parse_response_generation(self, response_text: str, email: EmailData, classification: EmailClassification) -> EmailResponse:
        """Parse AI response generation"""
        try:
            lines = response_text.strip().split('\n')
            result = {}
            current_section = None
            
            for line in lines:
                if line.startswith(('SUBJECT:', 'BODY:', 'TONE:', 'CONFIDENCE:', 'AUTO_SEND:', 'REASONING:')):
                    key, value = line.split(':', 1)
                    current_section = key.strip().lower()
                    result[current_section] = value.strip()
                elif current_section and line.strip():
                    result[current_section] += '\n' + line.strip()
            
            subject = result.get('subject', f"Re: {email.subject}")
            body = result.get('body', 'Thank you for your email. I will review and respond shortly.')
            tone = result.get('tone', 'professional')
            confidence = float(result.get('confidence', '0.7'))
            auto_send = result.get('auto_send', 'false').lower() == 'true'
            reasoning = result.get('reasoning', 'Standard response generated')
            
            # Auto-send decision based on confidence and threshold
            auto_send_threshold = self.config['processing']['auto_send_threshold']
            auto_send_recommended = auto_send and confidence >= auto_send_threshold
            
            return EmailResponse(
                subject=subject,
                body=body,
                tone=tone,
                confidence=confidence,
                auto_send_recommended=auto_send_recommended,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing response generation: {e}")
            return EmailResponse(
                subject=f"Re: {email.subject}",
                body="Thank you for your email. I will review and respond shortly.",
                tone="professional",
                confidence=0.5,
                auto_send_recommended=False,
                reasoning="Fallback response due to parsing error"
            )
    
    def send_email(self, to_email: str, subject: str, body: str, thread_id: str = None) -> bool:
        """Send email via Gmail API"""
        if not self.gmail_service:
            self.logger.error("Gmail service not initialized")
            return False
        
        try:
            # Create message
            message = MIMEText(body)
            message['to'] = to_email
            message['subject'] = subject
            
            # Add thread ID for replies
            raw_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
            if thread_id:
                raw_message['threadId'] = thread_id
            
            # Send message
            sent_message = self.gmail_service.users().messages().send(
                userId='me', body=raw_message
            ).execute()
            
            self.logger.info(f"Email sent successfully: {sent_message['id']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
            return False
    
    def create_draft(self, to_email: str, subject: str, body: str, thread_id: str = None) -> bool:
        """Create email draft via Gmail API"""
        if not self.gmail_service:
            self.logger.error("Gmail service not initialized")
            return False
        
        try:
            # Create message
            message = MIMEText(body)
            message['to'] = to_email
            message['subject'] = subject
            
            # Create draft
            draft_message = {'message': {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}}
            if thread_id:
                draft_message['message']['threadId'] = thread_id
            
            draft = self.gmail_service.users().drafts().create(
                userId='me', body=draft_message
            ).execute()
            
            self.logger.info(f"Draft created successfully: {draft['id']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating draft: {e}")
            return False
    
    def process_emails(self) -> Dict:
        """Main email processing workflow"""
        self.logger.info("Starting email processing...")
        
        results = {
            'processed': 0,
            'classified': {},
            'responses_generated': 0,
            'drafts_created': 0,
            'emails_sent': 0,
            'errors': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Get recent emails
            emails = self.get_recent_emails()
            
            for email in emails:
                try:
                    # Classify email
                    classification = self.classify_email(email)
                    
                    # Update stats
                    category = classification.category
                    if category not in results['classified']:
                        results['classified'][category] = 0
                    results['classified'][category] += 1
                    
                    self.stats['categories'][category] = self.stats['categories'].get(category, 0) + 1
                    
                    # Generate response if needed
                    if classification.requires_response:
                        response = self.generate_response(email, classification)
                        
                        if response:
                            results['responses_generated'] += 1
                            
                            # Extract sender email from "Name <email@domain.com>" format
                            sender_email = re.search(r'<(.+?)>', email.sender)
                            if sender_email:
                                to_email = sender_email.group(1)
                            else:
                                to_email = email.sender
                            
                            # Decide whether to send or draft
                            if response.auto_send_recommended:
                                if self.send_email(to_email, response.subject, response.body, email.thread_id):
                                    results['emails_sent'] += 1
                                    self.stats['auto_sent'] += 1
                            else:
                                if self.create_draft(to_email, response.subject, response.body, email.thread_id):
                                    results['drafts_created'] += 1
                    
                    results['processed'] += 1
                    self.stats['emails_processed'] += 1
                    
                    self.logger.info(f"Processed email: {email.subject[:50]}... - Category: {classification.category}")
                    
                except Exception as e:
                    error_msg = f"Error processing email {email.id}: {str(e)}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            # Update stats
            self.stats['responses_generated'] += results['responses_generated']
            self.stats['last_run'] = datetime.now().isoformat()
            
            self.logger.info(f"Email processing complete. Processed: {results['processed']}, "
                           f"Responses: {results['responses_generated']}, "
                           f"Sent: {results['emails_sent']}, "
                           f"Drafts: {results['drafts_created']}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error in email processing workflow: {str(e)}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
            return results
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()
    
    def test_setup(self) -> Dict:
        """Test the setup and return status"""
        status = {
            'gmail_api': False,
            'openai_api': False,
            'config_loaded': True,
            'credentials_found': False,
            'token_found': False,
            'errors': []
        }
        
        # Test Gmail API
        try:
            if self.gmail_service:
                profile = self.gmail_service.users().getProfile(userId='me').execute()
                status['gmail_api'] = True
                self.logger.info(f"Gmail API working. Email: {profile.get('emailAddress')}")
            else:
                status['errors'].append("Gmail service not initialized")
        except Exception as e:
            status['errors'].append(f"Gmail API error: {str(e)}")
        
        # Test OpenAI API
        try:
            if self.openai_client:
                # Simple test call
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=1
                )
                status['openai_api'] = True
                self.logger.info("OpenAI API working")
            else:
                status['errors'].append("OpenAI client not initialized")
        except Exception as e:
            status['errors'].append(f"OpenAI API error: {str(e)}")
        
        # Check files
        status['credentials_found'] = os.path.exists(self.config['gmail']['credentials_file'])
        status['token_found'] = os.path.exists(self.config['gmail']['token_file'])
        
        return status

if __name__ == "__main__":
    # Quick test
    assistant = EnhancedEmailAssistant()
    status = assistant.test_setup()
    
    print("🔍 Setup Status:")
    for key, value in status.items():
        if key != 'errors':
            icon = "✅" if value else "❌"
            print(f"{icon} {key}: {value}")
    
    if status['errors']:
        print("\n❌ Errors:")
        for error in status['errors']:
            print(f"  - {error}")
    
    if status['gmail_api'] and status['openai_api']:
        print("\n🚀 Ready to process emails!")
        # Uncomment to run processing
        # results = assistant.process_emails()
        # print(f"Results: {results}")
    else:
        print("\n⚠️ Setup incomplete. Please check configuration.")