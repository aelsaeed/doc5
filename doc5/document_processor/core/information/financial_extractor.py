"""
Financial entity extraction from document text
"""
import re
import logging

logger = logging.getLogger(__name__)

class FinancialEntityExtractor:
    """
    Financial entity extractor from document text
    """
    
    def __init__(self):
        """Initialize financial entity extractor"""
        logger.info("Financial entity extractor initialized")
        
        # Initialize regex patterns
        self.amount_pattern = r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD)'
        self.date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        self.account_pattern = r'\b(?:Account|Acct|A/C)(?:\s|:|\.|#)+\d+\b|\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b'
        self.entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|LLC|Corp|Corporation|Company|Co|Ltd)\b'
    
    def extract_entities(self, text):
        """
        Extract financial entities from text
        
        Args:
            text (str): Document text content
            
        Returns:
            list: List of extracted entities
        """
        entities = []
        
        # Extract amounts
        amounts = self._extract_amounts(text)
        entities.extend(amounts)
        
        # Extract dates
        dates = self._extract_dates(text)
        entities.extend(dates)
        
        # Extract account numbers
        accounts = self._extract_accounts(text)
        entities.extend(accounts)
        
        # Extract organization entities
        orgs = self._extract_organizations(text)
        entities.extend(orgs)
        
        return entities
    
    def _extract_amounts(self, text):
        """Extract amount entities from text"""
        entities = []
        for match in re.finditer(self.amount_pattern, text):
            entities.append({
                'type': 'AMOUNT',
                'text': match.group(0),
                'start_idx': match.start(),
                'end_idx': match.end()
            })
        return entities
    
    def _extract_dates(self, text):
        """Extract date entities from text"""
        entities = []
        for match in re.finditer(self.date_pattern, text):
            entities.append({
                'type': 'DATE',
                'text': match.group(0),
                'start_idx': match.start(),
                'end_idx': match.end()
            })
        return entities
    
    def _extract_accounts(self, text):
        """Extract account number entities from text"""
        entities = []
        for match in re.finditer(self.account_pattern, text):
            entities.append({
                'type': 'ACCOUNT',
                'text': match.group(0),
                'start_idx': match.start(),
                'end_idx': match.end()
            })
        return entities
    
    def _extract_organizations(self, text):
        """Extract organization entities from text"""
        entities = []
        for match in re.finditer(self.entity_pattern, text):
            entities.append({
                'type': 'ENTITY',
                'text': match.group(0),
                'start_idx': match.start(),
                'end_idx': match.end()
            })
        return entities