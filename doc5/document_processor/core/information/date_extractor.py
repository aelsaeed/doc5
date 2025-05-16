"""
Date extraction module for document processing
"""
import logging
from typing import List, Dict, Any
import spacy
from dateutil import parser

from document_processor.db.models import ImportantDate

logger = logging.getLogger(__name__)

class DateExtractor:
    """Extract important dates from documents"""
    
    # Event keywords for matching dates to specific events
    EVENT_KEYWORDS = {
        "Tax Return Submission": ["tax return due", "filing deadline", "submission date"],
        "Dividend Payment": ["dividend payment", "dividend date"],
        "Annual Meeting": ["annual meeting", "shareholder meeting", "annual general meeting"],
        "Filing Deadline": ["form due", "filing due", "deadline for filing"],
        "Payment Date": ["payment date", "distribution date", "payment will be made"],
        "Statement Date": ["statement date", "as of date", "reporting date", "report date"],
        "Effective Date": ["effective date", "effective as of", "takes effect"]
    }
    
    def __init__(self, nlp_model="en_core_web_sm"):
        """
        Initialize the date extractor
        
        Args:
            nlp_model (str): Name of the spaCy model to use
        """
        self.nlp = spacy.load(nlp_model)
        logger.info(f"Date extractor initialized with model: {nlp_model}")
    
    def extract_important_dates(self, text: str) -> List[ImportantDate]:
        """
        Extract dates associated with key events
        
        Args:
            text (str): Document text
            
        Returns:
            List[ImportantDate]: List of extracted important dates
        """
        doc = self.nlp(text)
        important_dates = []
        
        # Process each entity identified as a date
        for ent in doc.ents:
            if ent.label_ == "DATE":
                try:
                    # Try to parse the date into a standard format
                    date_str = parser.parse(ent.text).strftime("%Y-%m-%d")
                except ValueError:
                    # If parsing fails, use the original text
                    date_str = ent.text
                    continue  # Skip dates that can't be parsed to standard format
                
                # Get the surrounding sentence for context
                sentence = ent.sent.text.lower()
                
                # Try to match the sentence to an event type based on keywords
                event_type = self._identify_event_type(sentence)
                
                if event_type:
                    important_dates.append(ImportantDate(
                        event=event_type,
                        date=date_str
                    ))
        
        return important_dates
    
    def _identify_event_type(self, sentence: str) -> str:
        """
        Identify event type based on keywords in the sentence
        
        Args:
            sentence (str): Sentence containing the date
            
        Returns:
            str: Identified event type or "Other Date" if not matched
        """
        for event, keywords in self.EVENT_KEYWORDS.items():
            if any(keyword in sentence for keyword in keywords):
                return event
        
        # Check for any date-related terms to identify general dates
        general_date_terms = ["date", "due", "deadline", "by", "before", "after"]
        if any(term in sentence for term in general_date_terms):
            return "Other Date"
        
        return None
    
    def group_dates_by_event(self, dates: List[ImportantDate]) -> Dict[str, List[str]]:
        """
        Group dates by event type
        
        Args:
            dates (List[ImportantDate]): List of important dates
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping event types to lists of dates
        """
        grouped_dates = {}
        
        for date in dates:
            if date.event not in grouped_dates:
                grouped_dates[date.event] = []
            grouped_dates[date.event].append(date.date)
        
        return grouped_dates