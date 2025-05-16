"""
Targeted extraction module using doctr for locating keywords and LayoutLMv3 for extracting specific fields
"""
import os
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional
import re
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

# Import doctr
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False

# Import LayoutLMv3
try:
    from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ForTokenClassification
    LAYOUTLMV3_AVAILABLE = True
except ImportError:
    LAYOUTLMV3_AVAILABLE = False

from document_processor.utils.custom_exceptions import (
    FileTypeError, FileReadError, TextExtractionError, EmptyTextError
)
from document_processor.utils.gpu_utils import check_gpu_availability
from document_processor.utils.validation import validate_file

logger = logging.getLogger(__name__)

class TargetedExtractor:
    """
    Extractor that scans for specific keywords using doctr and extracts values using LayoutLMv3
    """
    
    # Define fields to extract for each document type
    DOCUMENT_FIELDS = {
        "K1 (Schedule K-1)": [
            "partner_name", "partner_address", "ein", "ordinary_income",
            "interest_income", "dividend_income", "royalty_income",
            "capital_gain", "tax_year", "guaranteed_payments"
        ],
        "W2 (Form W-2)": [
            "employee_name", "employee_ssn", "employer_name", "employer_ein",
            "wages", "federal_tax", "social_security_wages", "social_security_tax",
            "medicare_wages", "medicare_tax", "state_wages", "state_tax",
            "tax_year"
        ]
    }
    
    def __init__(self):
        """Initialize targeted extractor"""
        # Initialize device
        self.device = check_gpu_availability()
        logger.info(f"Targeted extractor initialized with device: {self.device}")
        
        # Initialize doctr if available
        self.doctr_model = None
        if DOCTR_AVAILABLE:
            try:
                logger.info("Initializing doctr OCR model")
                self.doctr_model = ocr_predictor(
                    det_arch='db_resnet50',
                    reco_arch='crnn_vgg16_bn',
                    pretrained=True
                )
                self.doctr_model.det_predictor.model = self.doctr_model.det_predictor.model.to(self.device)
                self.doctr_model.reco_predictor.model = self.doctr_model.reco_predictor.model.to(self.device)
                logger.info("doctr OCR model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize doctr OCR model: {str(e)}")
                self.doctr_model = None
        else:
            logger.warning("doctr not available. Install with 'pip install python-doctr'")
        
        # Initialize LayoutLMv3 if available
        self.tokenizer = None
        self.layoutlmv3_model = None
        if LAYOUTLMV3_AVAILABLE:
            try:
                logger.info("Initializing LayoutLMv3 model")
                self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained('microsoft/layoutlmv3-base')
                self.layoutlmv3_model = LayoutLMv3ForTokenClassification.from_pretrained('microsoft/layoutlmv3-base')
                self.layoutlmv3_model.to(self.device)
                logger.info("LayoutLMv3 model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LayoutLMv3 model: {str(e)}")
                self.layoutlmv3_model = None
        else:
            logger.warning("LayoutLMv3 not available. Install with 'pip install transformers'")
    
    def extract(self, file_path: str, doc_type: str = None) -> Dict[str, Any]:
        """
        Extract specific targeted information from document using doctr and LayoutLMv3
        
        Args:
            file_path (str): Path to document file
            doc_type (str, optional): Document type for targeted extraction
            
        Returns:
            Dict[str, Any]: Dictionary with extracted fields
        """
        try:
            # Validate file
            result, error = validate_file(file_path)
            if not result:
                raise error
            
            # Check if we have the required models
            if self.doctr_model is None:
                logger.warning("doctr model not available, using basic extraction")
                return self._extract_using_basic(file_path, doc_type)
            
            # Load document based on file type
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                doc = DocumentFile.from_pdf(file_path)
            else:
                doc = DocumentFile.from_images(file_path)
            
            # Process the document with doctr
            logger.info(f"Processing document: {file_path}")
            result = self.doctr_model(doc)
            
            # Save debug visualization of the document with bounding boxes
            debug_path = None
            if os.path.exists(file_path.replace('.pdf', '.jpg')) or file_path.endswith(('.jpg', '.jpeg', '.png')):
                # Use the image file directly if PDF was converted or it's already an image
                image_path = file_path.replace('.pdf', '.jpg') if file_path.endswith('.pdf') else file_path
                # Collect all words and bboxes
                all_words = []
                all_bboxes = []
                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            for word in line.words:
                                all_words.append(word.value)
                                all_bboxes.append(word.geometry)
                # Create debug visualization if words were extracted
                if all_words:
                    debug_path = self.visualize_bboxes(image_path, all_bboxes, all_words)
                    logger.info(f"Created bounding box visualization: {debug_path}")
            
            # Collect words and bounding boxes from all pages
            extracted_fields = {}
            for page_idx, page in enumerate(result.pages):
                logger.info(f"Processing page {page_idx + 1} of document: {file_path}")
                
                # Collect words and bounding boxes for this page
                words = []
                bboxes = []
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            words.append(word.value)
                            # Correctly unpack the geometry attribute
                            bboxes.append(word.geometry)
                
                if not words:
                    logger.warning(f"No text extracted from page {page_idx + 1}")
                    continue
                
                # Extract fields from this page
                page_fields = self._extract_fields_from_page(words, bboxes, doc_type)
                
                # Merge fields from this page
                for field, value in page_fields.items():
                    if field not in extracted_fields or not extracted_fields[field] or extracted_fields[field] == "Not found":
                        extracted_fields[field] = value
            
            # Ensure all expected fields are present, even if not found
            if doc_type in self.DOCUMENT_FIELDS:
                for field in self.DOCUMENT_FIELDS[doc_type]:
                    if field not in extracted_fields:
                        extracted_fields[field] = "Not found"
            
            # Add debug info if visualization was created
            if debug_path:
                extracted_fields["_debug_visualization"] = debug_path
            
            return {
                "extracted_fields": extracted_fields,
                "document_type": doc_type
            }
            
        except (FileTypeError, FileReadError) as e:
            raise
        except Exception as e:
            logger.error(f"Error in targeted extraction for {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise TextExtractionError(file_path, f"Targeted extraction failed: {str(e)}")
    
    def _extract_fields_from_page(self, words, bboxes, doc_type):
        """
        Extract fields from a page using a combination of pattern matching and spatial analysis
        
        Args:
            words (List[str]): List of words extracted by doctr
            bboxes (List[Tuple]): Bounding boxes for each word
            doc_type (str): Document type
            
        Returns:
            Dict[str, str]: Dictionary of extracted fields
        """
        extracted_fields = {}
        
        # Create a combined text for pattern matching
        combined_text = " ".join(words)
        logger.debug(f"Combined text: {combined_text[:200]}...")
        
        # Create a spatial map of words with their positions
        word_map = []
        for i, (word, bbox) in enumerate(zip(words, bboxes)):
            # Calculate center point
            (x0, y0), (x1, y1) = bbox
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            word_map.append({
                "word": word,
                "bbox": bbox,
                "center": (center_x, center_y),
                "index": i
            })
        
        # Extract based on document type
        if doc_type == "W2 (Form W-2)":
            extracted_fields.update(self._extract_w2_fields(words, word_map, combined_text))
        elif doc_type == "K1 (Schedule K-1)":
            extracted_fields.update(self._extract_k1_fields(words, word_map, combined_text))
        
        return extracted_fields
    
    def _extract_w2_fields(self, words, word_map, combined_text):
        """
        Extract fields specific to W-2 forms
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Spatial map of words
            combined_text (str): Combined text
            
        Returns:
            Dict[str, str]: Dictionary of extracted fields
        """
        extracted_fields = {}
        
        # Extract Employee SSN
        extracted_fields["employee_ssn"] = self._extract_ssn(words, word_map, combined_text)
        
        # Extract Employee Name
        extracted_fields["employee_name"] = self._extract_employee_name(words, word_map, combined_text)
        
        # Extract Employer Name
        extracted_fields["employer_name"] = self._extract_employer_name(words, word_map, combined_text)
        
        # Extract Employer EIN
        extracted_fields["employer_ein"] = self._extract_employer_ein(words, word_map, combined_text)
        
        # Extract Box values
        # Box 1: Wages, tips, other compensation
        extracted_fields["wages"] = self._extract_box_value(words, word_map, combined_text, "1", 
                                                           ["wages", "tips", "compensation"])
        
        # Box 2: Federal income tax withheld
        extracted_fields["federal_tax"] = self._extract_box_value(words, word_map, combined_text, "2", 
                                                                ["federal", "income", "tax", "withheld"])
        
        # Box 3: Social security wages
        extracted_fields["social_security_wages"] = self._extract_box_value(words, word_map, combined_text, "3", 
                                                                           ["social", "security", "wages"])
        
        # Box 4: Social security tax withheld
        extracted_fields["social_security_tax"] = self._extract_box_value(words, word_map, combined_text, "4", 
                                                                         ["social", "security", "tax"])
        
        # Box 5: Medicare wages and tips
        extracted_fields["medicare_wages"] = self._extract_box_value(words, word_map, combined_text, "5", 
                                                                    ["medicare", "wages", "tips"])
        
        # Box 6: Medicare tax withheld
        extracted_fields["medicare_tax"] = self._extract_box_value(words, word_map, combined_text, "6", 
                                                                 ["medicare", "tax"])
        
        # Box 16: State wages, tips, etc.
        extracted_fields["state_wages"] = self._extract_box_value(words, word_map, combined_text, "16", 
                                                                ["state", "wages"])
        
        # Box 17: State income tax
        extracted_fields["state_tax"] = self._extract_box_value(words, word_map, combined_text, "17", 
                                                              ["state", "income", "tax"])
        
        # Extract Tax Year
        extracted_fields["tax_year"] = self._extract_tax_year(words, word_map, combined_text)
        
        return extracted_fields
    
    def _extract_k1_fields(self, words, word_map, combined_text):
        """
        Extract fields specific to K-1 forms
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Spatial map of words
            combined_text (str): Combined text
            
        Returns:
            Dict[str, str]: Dictionary of extracted fields
        """
        extracted_fields = {}
        
        # Extract Partner Name
        extracted_fields["partner_name"] = self._extract_partner_name(words, word_map, combined_text)
        
        # Extract Partner Address
        extracted_fields["partner_address"] = self._extract_partner_address(words, word_map, combined_text)
        
        # Extract EIN
        extracted_fields["ein"] = self._extract_ein(words, word_map, combined_text)
        
        # Extract Income Fields
        # For these fields we'll use a more generic approach to extract monetary values by type
        
        # Ordinary Income
        extracted_fields["ordinary_income"] = self._extract_income_field(words, word_map, combined_text, 
                                                                        ["ordinary", "income"])
        
        # Interest Income
        extracted_fields["interest_income"] = self._extract_income_field(words, word_map, combined_text, 
                                                                        ["interest", "income"])
        
        # Dividend Income
        extracted_fields["dividend_income"] = self._extract_income_field(words, word_map, combined_text, 
                                                                        ["dividend", "income"])
        
        # Royalty Income
        extracted_fields["royalty_income"] = self._extract_income_field(words, word_map, combined_text, 
                                                                       ["royalty", "income"])
        
        # Capital Gain
        extracted_fields["capital_gain"] = self._extract_income_field(words, word_map, combined_text, 
                                                                     ["capital", "gain"])
        
        # Guaranteed Payments
        extracted_fields["guaranteed_payments"] = self._extract_income_field(words, word_map, combined_text, 
                                                                           ["guaranteed", "payments"])
        
        # Extract Tax Year
        extracted_fields["tax_year"] = self._extract_tax_year(words, word_map, combined_text)
        
        return extracted_fields
    
    def _extract_ssn(self, words, word_map, combined_text):
        """
        Extract SSN using multiple strategies
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Spatial map of words
            combined_text (str): Combined text
            
        Returns:
            str: Extracted SSN or None
        """
        # Strategy 1: Look for SSN pattern (XXX-XX-XXXX)
        ssn_pattern = r'(\d{3}-\d{2}-\d{4})'
        ssn_match = re.search(ssn_pattern, combined_text)
        if ssn_match:
            return ssn_match.group(1)
        
        # Strategy 2: Look for SSN near context words
        context_words = ["SSN", "social", "security", "number"]
        for i, word in enumerate(words):
            if any(context.lower() in word.lower() for context in context_words):
                # Check nearby words for SSN pattern
                for j in range(max(0, i-5), min(len(words), i+6)):
                    if re.match(r'\d{3}-\d{2}-\d{4}', words[j]):
                        return words[j]
        
        # Strategy 3: Look for any 9-digit number with or without hyphens
        nine_digit_pattern = r'(\d{3}[\s-]?\d{2}[\s-]?\d{4})'
        nine_digit_match = re.search(nine_digit_pattern, combined_text)
        if nine_digit_match:
            # Format with hyphens if not present
            ssn = nine_digit_match.group(1)
            if '-' not in ssn:
                ssn = re.sub(r'(\d{3})[\s]?(\d{2})[\s]?(\d{4})', r'\1-\2-\3', ssn)
            return ssn
        
        return None
    
    def _extract_employee_name(self, words, word_map, combined_text):
        """
        Extract employee name using multiple strategies
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Spatial map of words
            combined_text (str): Combined text
            
        Returns:
            str: Extracted employee name or None
        """
        # Strategy 1: Look for common name patterns
        name_pattern = r'[A-Z][a-z]+\s+[A-Z](?:\.?\s+)[A-Z][A-Z]+'
        name_match = re.search(name_pattern, combined_text)
        if name_match:
            return name_match.group(0)
        
        # Strategy 2: Look for "Employee's name" or "Employee name" label
        for i, word in enumerate(words):
            if "employee" in word.lower() and "name" in (words[i+1].lower() if i+1 < len(words) else ""):
                # Find the next capitalized words that could be a name
                for j in range(i+2, min(i+10, len(words))):
                    if words[j][0].isupper() and len(words[j]) > 1:
                        # Check if we have a pattern like "FirstName MI LastName"
                        if j+2 < len(words) and len(words[j+1]) == 1 and words[j+2][0].isupper():
                            return f"{words[j]} {words[j+1]} {words[j+2]}"
                        # Check if we have a pattern like "FirstName LastName"
                        elif j+1 < len(words) and words[j+1][0].isupper():
                            return f"{words[j]} {words[j+1]}"
        
        # Strategy 3: Look for a pattern of first name, middle initial, last name based on capitalization
        for i in range(len(words)-2):
            if (words[i][0].isupper() and words[i].isalpha() and 
                len(words[i+1]) == 1 and words[i+1].isupper() and
                words[i+2][0].isupper() and words[i+2].isalpha()):
                return f"{words[i]} {words[i+1]} {words[i+2]}"
        
        return None
    
    def _extract_employer_name(self, words, word_map, combined_text):
        """
        Extract employer name using multiple strategies
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Spatial map of words
            combined_text (str): Combined text
            
        Returns:
            str: Extracted employer name or None
        """
        # Strategy 1: Look for "Employer name" or "Employer's name" label
        for i, word in enumerate(words):
            if "employer" in word.lower() and "name" in (words[i+1].lower() if i+1 < len(words) else ""):
                # Find the next capitalized words that could be a company name
                company_name_parts = []
                for j in range(i+2, min(i+10, len(words))):
                    if words[j][0].isupper() and len(words[j]) > 1:
                        company_name_parts.append(words[j])
                        # Stop if we hit a non-capitalized word or reach a limit
                        if len(company_name_parts) >= 3 or (j+1 < len(words) and not words[j+1][0].isupper()):
                            break
                
                if company_name_parts:
                    return " ".join(company_name_parts)
        
        # Strategy 2: Look for company indicators
        company_indicators = ["Inc", "LLC", "Corp", "Company", "Co", "Ltd"]
        for i, word in enumerate(words):
            if any(indicator in word for indicator in company_indicators):
                # Get preceding words that might be part of the company name
                company_name_parts = []
                # Look backward for capitalized words that might be part of company name
                for j in range(i, max(i-5, -1), -1):
                    if words[j][0].isupper():
                        company_name_parts.insert(0, words[j])
                    else:
                        break
                
                if company_name_parts:
                    company_name_parts.append(word)  # Add the indicator word
                    return " ".join(company_name_parts)
        
        # Strategy 3: Look for common patterns like "The Big Company"
        for i, word in enumerate(words):
            if word.lower() == "the" and i+2 < len(words):
                if words[i+1][0].isupper() and words[i+2][0].isupper():
                    potential_name = f"{word} {words[i+1]} {words[i+2]}"
                    # Check next word for company indicator
                    if i+3 < len(words) and any(indicator in words[i+3] for indicator in company_indicators):
                        potential_name += f" {words[i+3]}"
                    return potential_name
        
        return None
    
    def _extract_employer_ein(self, words, word_map, combined_text):
        """
        Extract employer EIN using multiple strategies
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Spatial map of words
            combined_text (str): Combined text
            
        Returns:
            str: Extracted employer EIN or None
        """
        # Look for EIN format (XX-XXXXXXX)
        ein_pattern = r'(\d{2}-\d{7})'
        ein_matches = re.findall(ein_pattern, combined_text)
        
        if ein_matches:
            # Strategy 1: Look for EIN near context words
            context_words = ["EIN", "employer", "identification", "number"]
            for i, word in enumerate(words):
                if any(context.lower() in word.lower() for context in context_words):
                    # Check nearby words for EIN pattern
                    for j in range(max(0, i-5), min(len(words), i+6)):
                        if re.match(r'\d{2}-\d{7}', words[j]):
                            return words[j]
            
            # Strategy 2: If multiple EINs, look for position in standard W2 layout
            if len(ein_matches) > 1:
                # Usually employer EIN is near the top of the document
                for word_info in word_map[:20]:  # Check first 20 words
                    if re.match(r'\d{2}-\d{7}', word_info["word"]):
                        return word_info["word"]
            
            # Strategy 3: If all else fails, return the first match
            return ein_matches[0]
        
        return None
    
    def _extract_box_value(self, words, word_map, combined_text, box_number, context_keywords):
        """
        Extract value from a specific box on a form using multiple strategies
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Spatial map of words
            combined_text (str): Combined text
            box_number (str): Box number (e.g., "1", "2")
            context_keywords (List[str]): Keywords associated with this box
            
        Returns:
            str: Extracted value or None
        """
        # Strategy 1: Find box number and look to the right for a numeric value
        box_indicators = [w for w in word_map if w["word"] == box_number]
        
        for box in box_indicators:
            # Get words to the right of the box number, sorted by distance
            box_center = box["center"]
            right_words = sorted(
                [w for w in word_map if w["center"][0] > box_center[0]],
                key=lambda w: (abs(w["center"][1] - box_center[1])*3 + (w["center"][0] - box_center[0]))
            )
            
            # Check first few words to the right for monetary values
            for word_info in right_words[:5]:
                # Look for currency format or plain number
                if re.match(r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?', word_info["word"]):
                    return word_info["word"].replace('$', '')
        
        # Strategy 2: Look for box label with context keywords
        for keyword in context_keywords:
            keyword_matches = [w for w in word_map if keyword.lower() in w["word"].lower()]
            for keyword_match in keyword_matches:
                # Get words nearby, prioritizing those to the right and at similar vertical position
                keyword_center = keyword_match["center"]
                nearby_words = sorted(
                    [w for w in word_map if w != keyword_match],
                    key=lambda w: ((w["center"][0] - keyword_center[0] if w["center"][0] > keyword_center[0] else 5) +
                                  abs(w["center"][1] - keyword_center[1])*3)
                )
                
                # Check nearby words for monetary values
                for word_info in nearby_words[:7]:
                    if re.match(r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?', word_info["word"]):
                        return word_info["word"].replace('$', '')
        
        # Strategy 3: Use regex pattern to find box number with value
        box_pattern = fr'(?:box\s*)?{box_number}\s*[^0-9]*?(\$?\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{2}})?)'
        box_match = re.search(box_pattern, combined_text.lower())
        if box_match:
            return box_match.group(1).replace('$', '')
        
        # Strategy 4: Look for context keywords with values
        for keyword in context_keywords:
            pattern = fr'{keyword}[^0-9]*?(\$?\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{2}})?)'
            match = re.search(pattern, combined_text.lower())
            if match:
                return match.group(1).replace('$', '')
        
        return None
    
    def _extract_tax_year(self, words, word_map, combined_text):
        """
        Extract tax year from the document
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Spatial map of words
            combined_text (str): Combined text
            
        Returns:
            str: Extracted tax year or None
        """
        # Strategy 1: Look for "Tax Year" label
        year_pattern = r'tax\s+year[^0-9]*?(20\d{2})'
        year_match = re.search(year_pattern, combined_text.lower())
        if year_match:
            return year_match.group(1)
        
        # Strategy 2: Look for a standalone 4-digit year (20XX)
        for word in words:
            if re.match(r'^20\d{2}$', word):
                return word
        
        # Strategy 3: Look for year in header or title
        year_pattern = r'form\s+w-?2(?:[^0-9]*)(20\d{2})'
        year_match = re.search(year_pattern, combined_text.lower())
        if year_match:
            return year_match.group(1)
        
        # Strategy 4: Use any year-like pattern as fallback
        year_pattern = r'(20\d{2})'
        year_match = re.search(year_pattern, combined_text)
        if year_match:
            return year_match.group(1)
        
        return None
    
    def _extract_partner_name(self, words, word_map, combined_text):
        """
        Extract partner name from K-1 form
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Spatial map of words
            combined_text (str): Combined text
            
        Returns:
            str: Extracted partner name or None
        """
        # Strategy 1: Look for "Partner's name" or similar label
        for i, word in enumerate(words):
            if "partner" in word.lower() and "name" in (words[i+1].lower() if i+1 < len(words) else ""):
                # Find the next capitalized words that could be a name
                for j in range(i+2, min(i+10, len(words))):
                    if words[j][0].isupper() and len(words[j]) > 1:
                        # Look for 1-3 consecutive capitalized words that might form a name
                        name_parts = [words[j]]
                        for k in range(j+1, min(j+3, len(words))):
                            if words[k][0].isupper():
                                name_parts.append(words[k])
                            else:
                                break
                        return " ".join(name_parts)
        
        # Strategy 2: Look for "Part II" section (usually contains partner info)
        for i, word in enumerate(words):
            if word.lower() == "part" and i+1 < len(words) and words[i+1] in ["II", "2"]:
                # Look for capitalized words following this section
                for j in range(i+2, min(i+15, len(words))):
                    if words[j][0].isupper() and len(words[j]) > 1:
                        # Check if we have a pattern of 2-3 capitalized words
                        if j+1 < len(words) and words[j+1][0].isupper():
                            if j+2 < len(words) and words[j+2][0].isupper():
                                return f"{words[j]} {words[j+1]} {words[j+2]}"
                            return f"{words[j]} {words[j+1]}"
                        return words[j]
        
        # Strategy 3: Look for common name patterns
        for i in range(len(words)-1):
            if words[i][0].isupper() and words[i].isalpha() and len(words[i]) > 1:
                if i+1 < len(words) and words[i+1][0].isupper() and words[i+1].isalpha():
                    # Check if we have a company name pattern
                    if any(indicator in words[i+1] for indicator in ["Inc", "LLC", "Corp", "Company"]):
                        return f"{words[i]} {words[i+1]}"
                    # Check for individual name pattern
                    elif len(words[i]) <= 15 and len(words[i+1]) <= 15:  # Reasonable name lengths
                        return f"{words[i]} {words[i+1]}"
        
        return None
    
    def _extract_partner_address(self, words, word_map, combined_text):
        """
        Extract partner address from K-1 form
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Spatial map of words
            combined_text (str): Combined text
            
        Returns:
            str: Extracted partner address or None
        """
        # Strategy 1: Look for "Partner's address" or similar label
        for i, word in enumerate(words):
            if "partner" in word.lower() and "address" in (words[i+1].lower() if i+1 < len(words) else ""):
                # Address typically spans multiple words
                address_parts = []
                # Collect words that might be part of the address
                for j in range(i+2, min(i+15, len(words))):
                    # Stop at likely end of address patterns
                    if any(stopper in words[j].lower() for stopper in ["foreign", "domestic", "ein", "ssn"]):
                        break
                    # Add word to address parts
                    address_parts.append(words[j])
                    # Stop if we hit a pattern that suggests end of address (ZIP code)
                    if re.match(r'\d{5}(?:-\d{4})?', words[j]):
                        break
                
                if address_parts:
                    return " ".join(address_parts)
        
        # Strategy 2: Look for address pattern (street, city, state ZIP)
        address_pattern = r'([0-9]+\s+[A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\.?(?:,?\s+[A-Za-z]+(?:,?\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?))?)'
        address_match = re.search(address_pattern, combined_text)
        if address_match:
            return address_match.group(1)
        
        # Strategy 3: Look for "Part II" section and find address-like text
        for i, word in enumerate(words):
            if word.lower() == "part" and i+1 < len(words) and words[i+1] in ["II", "2"]:
                # After finding partner name, the address often follows
                for j in range(i+5, min(i+20, len(words))):
                    # Look for numeric start of street address
                    if re.match(r'^\d+$', words[j]) and j+1 < len(words):
                        address_parts = [words[j]]
                        # Collect following words until likely end of address
                        for k in range(j+1, min(j+10, len(words))):
                            address_parts.append(words[k])
                            # Stop at ZIP code pattern
                            if re.match(r'\d{5}(?:-\d{4})?', words[k]):
                                break
                        return " ".join(address_parts)
        
        return None
    
    def _extract_ein(self, words, word_map, combined_text):
        """
        Extract EIN from K-1 form (similar to employer_ein but with K-1 specific context)
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Spatial map of words
            combined_text (str): Combined text
            
        Returns:
            str: Extracted EIN or None
        """
        # Strategy 1: Look for EIN format (XX-XXXXXXX)
        ein_pattern = r'(\d{2}-\d{7})'
        ein_matches = re.findall(ein_pattern, combined_text)
        
        if ein_matches:
            # Look for EIN near context words
            context_words = ["EIN", "identification", "number", "employer"]
            for i, word in enumerate(words):
                if any(context.lower() in word.lower() for context in context_words):
                    # Check nearby words for EIN pattern
                    for j in range(max(0, i-3), min(len(words), i+4)):
                        if re.match(r'\d{2}-\d{7}', words[j]):
                            return words[j]
            
            # If multiple EINs found, use position heuristics
            if len(ein_matches) > 1:
                # K-1 typically has the partnership EIN near the top
                top_ein = None
                for word_info in word_map[:30]:  # Look in first 30 words
                    if re.match(r'\d{2}-\d{7}', word_info["word"]):
                        top_ein = word_info["word"]
                        break
                
                if top_ein:
                    return top_ein
            
            # Return first match if no better option
            return ein_matches[0]
        
        return None
    
    def _extract_income_field(self, words, word_map, combined_text, context_keywords):
        """
        Extract income field value from K-1 form
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Spatial map of words
            combined_text (str): Combined text
            context_keywords (List[str]): Keywords associated with this income type
            
        Returns:
            str: Extracted income value or None
        """
        # Strategy 1: Look for line item with context keywords and associated value
        for keyword in context_keywords:
            for i, word in enumerate(words):
                if keyword.lower() in word.lower():
                    # Look for monetary values nearby
                    for j in range(max(0, i-3), min(len(words), i+7)):
                        if re.match(r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?', words[j]):
                            return words[j].replace('$', '')
        
        # Strategy 2: Look for text pattern combining keywords and numbers
        keyword_pattern = fr'({"|".join(context_keywords)})[^0-9]*(-?\$?\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{2}})?)'
        keyword_match = re.search(keyword_pattern, combined_text.lower())
        if keyword_match:
            return keyword_match.group(2).replace('$', '')
        
        # Strategy 3: Look for a line number pattern (common in K-1 forms)
        line_patterns = {
            "ordinary_income": r'(?:line\s*1|1\.\s*ordinary)[^0-9]*(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            "interest_income": r'(?:line\s*5|5\.\s*interest)[^0-9]*(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            "dividend_income": r'(?:line\s*6|6[a-b]?\.\s*dividend)[^0-9]*(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            "royalty_income": r'(?:line\s*7|7\.\s*royalty)[^0-9]*(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            "capital_gain": r'(?:line\s*9|9[a-b]?\.\s*capital)[^0-9]*(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            "guaranteed_payments": r'(?:line\s*4|4[a-c]?\.\s*guaranteed)[^0-9]*(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        }
        
        # Find the appropriate line pattern for this field
        field_key = "_".join(context_keywords)
        if field_key in line_patterns:
            line_match = re.search(line_patterns[field_key], combined_text.lower())
            if line_match:
                return line_match.group(1).replace('$', '')
        
        return None
    
    def _extract_using_basic(self, file_path: str, doc_type: str = None) -> Dict[str, Any]:
        """
        Fallback extraction using regex patterns when doctr is not available
        
        Args:
            file_path (str): Path to document file
            doc_type (str, optional): Document type for targeted extraction
            
        Returns:
            Dict[str, Any]: Dictionary with extracted fields
        """
        try:
            import fitz  # PyMuPDF
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            
            if not text.strip():
                raise EmptyTextError(file_path)
            
            # Log full text for debugging
            logger.debug(f"Basic extraction text: {text[:200]}...")
            
            extracted_data = {}
            
            # Extract based on document type
            if doc_type == "W2 (Form W-2)":
                extracted_data = self._extract_w2_using_basic(text)
            elif doc_type == "K1 (Schedule K-1)":
                extracted_data = self._extract_k1_using_basic(text)
            
            # Ensure all expected fields are present
            if doc_type in self.DOCUMENT_FIELDS:
                for field in self.DOCUMENT_FIELDS[doc_type]:
                    if field not in extracted_data:
                        extracted_data[field] = "Not found"
            
            return {
                "extracted_fields": extracted_data,
                "document_type": doc_type
            }
                
        except Exception as e:
            logger.error(f"Error in basic extraction for {file_path}: {str(e)}")
            raise TextExtractionError(file_path, f"Basic extraction failed: {str(e)}")
    
    def _extract_w2_using_basic(self, text):
        """
        Extract W-2 fields using regex patterns from text
        
        Args:
            text (str): Document text
            
        Returns:
            Dict[str, str]: Dictionary of extracted fields
        """
        extracted_data = {}
        
        # Employee SSN
        ssn_pattern = r'(\d{3}-\d{2}-\d{4})'
        ssn_match = re.search(ssn_pattern, text)
        if ssn_match:
            extracted_data["employee_ssn"] = ssn_match.group(1)
        
        # Employee Name
        # Look for name patterns with surrounding context
        name_patterns = [
            r'Employee(?:\'s)?\s+name[^\n]*?([A-Z][a-z]+\s+[A-Z](?:\.|[a-z]+)?\s+[A-Z][A-Za-z]+)',
            r'([A-Z][a-z]+\s+[A-Z]\s+[A-Z]{3,})'  # Pattern like "Jane A DOE"
        ]
        
        for pattern in name_patterns:
            name_match = re.search(pattern, text, re.IGNORECASE)
            if name_match:
                extracted_data["employee_name"] = name_match.group(1)
                break
        
        # Employer Name
        # Look for employer name patterns
        employer_patterns = [
            r'Employer(?:\'s)?\s+name[^\n]*?([A-Z][A-Za-z\s]+(?:Inc\.?|LLC|Corp\.?|Company|Co\.?))',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+\s+Company)'  # Pattern like "The Big Company"
        ]
        
        for pattern in employer_patterns:
            employer_match = re.search(pattern, text, re.IGNORECASE)
            if employer_match:
                extracted_data["employer_name"] = employer_match.group(1)
                break
        
        # Employer EIN
        ein_pattern = r'(?:EIN|Employer\s+identification\s+number)[^\n]*?(\d{2}-\d{7})'
        ein_match = re.search(ein_pattern, text, re.IGNORECASE)
        if ein_match:
            extracted_data["employer_ein"] = ein_match.group(1)
        
        # Box fields with improved patterns
        
        # Box 1: Wages
        wages_pattern = r'(?:1\s+Wages|Wages,\s+tips)[^\n]*?(\d{1,3}(?:,\d{3})*\.\d{2})'
        wages_match = re.search(wages_pattern, text, re.IGNORECASE)
        if wages_match:
            extracted_data["wages"] = wages_match.group(1)
        
        # Box 2: Federal tax
        federal_tax_pattern = r'(?:2\s+Federal|Federal\s+income\s+tax)[^\n]*?(\d{1,3}(?:,\d{3})*\.\d{2})'
        federal_tax_match = re.search(federal_tax_pattern, text, re.IGNORECASE)
        if federal_tax_match:
            extracted_data["federal_tax"] = federal_tax_match.group(1)
        
        # Box 3: Social security wages
        ss_wages_pattern = r'(?:3\s+Social|Social\s+security\s+wages)[^\n]*?(\d{1,3}(?:,\d{3})*\.\d{2})'
        ss_wages_match = re.search(ss_wages_pattern, text, re.IGNORECASE)
        if ss_wages_match:
            extracted_data["social_security_wages"] = ss_wages_match.group(1)
        
        # Box 4: Social security tax
        ss_tax_pattern = r'(?:4\s+Social|Social\s+security\s+tax)[^\n]*?(\d{1,3}(?:,\d{3})*\.\d{2})'
        ss_tax_match = re.search(ss_tax_pattern, text, re.IGNORECASE)
        if ss_tax_match:
            extracted_data["social_security_tax"] = ss_tax_match.group(1)
        
        # Box 5: Medicare wages
        medicare_wages_pattern = r'(?:5\s+Medicare|Medicare\s+wages)[^\n]*?(\d{1,3}(?:,\d{3})*\.\d{2})'
        medicare_wages_match = re.search(medicare_wages_pattern, text, re.IGNORECASE)
        if medicare_wages_match:
            extracted_data["medicare_wages"] = medicare_wages_match.group(1)
        
        # Box 6: Medicare tax
        medicare_tax_pattern = r'(?:6\s+Medicare|Medicare\s+tax)[^\n]*?(\d{1,3}(?:,\d{3})*\.\d{2})'
        medicare_tax_match = re.search(medicare_tax_pattern, text, re.IGNORECASE)
        if medicare_tax_match:
            extracted_data["medicare_tax"] = medicare_tax_match.group(1)
        
        # Box 16: State wages
        state_wages_pattern = r'(?:16\s+State|State\s+wages)[^\n]*?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        state_wages_match = re.search(state_wages_pattern, text, re.IGNORECASE)
        if state_wages_match:
            extracted_data["state_wages"] = state_wages_match.group(1)
        
        # Box 17: State tax
        state_tax_pattern = r'(?:17\s+State|State\s+income\s+tax)[^\n]*?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        state_tax_match = re.search(state_tax_pattern, text, re.IGNORECASE)
        if state_tax_match:
            extracted_data["state_tax"] = state_tax_match.group(1)
        
        # Tax Year
        year_pattern = r'(20\d{2})'
        year_matches = re.findall(year_pattern, text)
        if year_matches:
            # Use the most frequently occurring year
            year_counts = Counter(year_matches)
            extracted_data["tax_year"] = year_counts.most_common(1)[0][0]
        
        return extracted_data
    
    def _extract_k1_using_basic(self, text):
        """
        Extract K-1 fields using regex patterns from text
        
        Args:
            text (str): Document text
            
        Returns:
            Dict[str, str]: Dictionary of extracted fields
        """
        extracted_data = {}
        
        # Partner Name
        partner_name_pattern = r'(?:Partner\'s\s+name|Part\s+II[^\n]*?partner)[^\n]*?([A-Z][a-zA-Z\s]+(?:LLC|Inc\.?|Corp\.?|Company)?)'
        partner_name_match = re.search(partner_name_pattern, text, re.IGNORECASE)
        if partner_name_match:
            extracted_data["partner_name"] = partner_name_match.group(1)
        
        # Partner Address
        partner_address_pattern = r'(?:Partner\'s\s+address|address\s+of\s+partner)[^\n]*?([0-9]+[^\n]*?(?:[A-Z]{2}\s+\d{5}(?:-\d{4})?))'
        partner_address_match = re.search(partner_address_pattern, text, re.IGNORECASE)
        if partner_address_match:
            extracted_data["partner_address"] = partner_address_match.group(1)
        
        # EIN
        ein_pattern = r'(?:Employer\s+identification\s+number|EIN)[^\n]*?(\d{2}-\d{7})'
        ein_match = re.search(ein_pattern, text, re.IGNORECASE)
        if ein_match:
            extracted_data["ein"] = ein_match.group(1)
        
        # Income fields - Using line numbers and keywords
        
        # Ordinary Income
        ordinary_income_pattern = r'(?:Ordinary\s+business\s+income|Line\s+1[^\n]*?ordinary)[^\n]*?(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ordinary_income_match = re.search(ordinary_income_pattern, text, re.IGNORECASE)
        if ordinary_income_match:
            extracted_data["ordinary_income"] = ordinary_income_match.group(1).replace('$', '')
        
        # Interest Income
        interest_income_pattern = r'(?:Interest\s+income|Line\s+5[^\n]*?interest)[^\n]*?(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        interest_income_match = re.search(interest_income_pattern, text, re.IGNORECASE)
        if interest_income_match:
            extracted_data["interest_income"] = interest_income_match.group(1).replace('$', '')
        
        # Dividend Income
        dividend_income_pattern = r'(?:Dividend\s+income|Line\s+6[a-b]?[^\n]*?dividend)[^\n]*?(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        dividend_income_match = re.search(dividend_income_pattern, text, re.IGNORECASE)
        if dividend_income_match:
            extracted_data["dividend_income"] = dividend_income_match.group(1).replace('$', '')
        
        # Royalty Income
        royalty_income_pattern = r'(?:Royalty\s+income|Line\s+7[^\n]*?royalty)[^\n]*?(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        royalty_income_match = re.search(royalty_income_pattern, text, re.IGNORECASE)
        if royalty_income_match:
            extracted_data["royalty_income"] = royalty_income_match.group(1).replace('$', '')
        
        # Capital Gain
        capital_gain_pattern = r'(?:Net\s+(?:short|long)-term\s+capital\s+gain|Line\s+9[a-c]?[^\n]*?capital)[^\n]*?(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        capital_gain_match = re.search(capital_gain_pattern, text, re.IGNORECASE)
        if capital_gain_match:
            extracted_data["capital_gain"] = capital_gain_match.group(1).replace('$', '')
        
        # Guaranteed Payments
        guaranteed_payments_pattern = r'(?:Guaranteed\s+payments|Line\s+4[^\n]*?guaranteed)[^\n]*?(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        guaranteed_payments_match = re.search(guaranteed_payments_pattern, text, re.IGNORECASE)
        if guaranteed_payments_match:
            extracted_data["guaranteed_payments"] = guaranteed_payments_match.group(1).replace('$', '')
        
        # Tax Year
        year_pattern = r'(?:tax\s+year|beginning|ended|ending|year\s+ended)[^\n]*?(20\d{2})'
        year_match = re.search(year_pattern, text, re.IGNORECASE)
        if year_match:
            extracted_data["tax_year"] = year_match.group(1)
        else:
            # Fallback to any year found
            year_pattern = r'(20\d{2})'
            year_matches = re.findall(year_pattern, text)
            if year_matches:
                # Use the most frequently occurring year
                year_counts = Counter(year_matches)
                extracted_data["tax_year"] = year_counts.most_common(1)[0][0]
        
        return extracted_data
    
    def visualize_bboxes(self, image_path, bboxes, words):
        """
        Visualize bounding boxes on the image for debugging
        
        Args:
            image_path (str): Path to the image
            bboxes (List): List of bounding boxes
            words (List[str]): List of words corresponding to bounding boxes
            
        Returns:
            str: Path to the debug image
        """
        try:
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # Get image dimensions
            width, height = img.size
            
            # Draw bounding boxes and words
            for bbox, word in zip(bboxes, words):
                (x0, y0), (x1, y1) = bbox
                # Convert normalized coordinates to pixel coordinates
                x0, y0 = int(x0 * width), int(y0 * height)
                x1, y1 = int(x1 * width), int(y1 * height)
                
                # Draw rectangle
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                
                # Draw word above rectangle
                draw.text((x0, y0-10), word, fill="red")
            
            # Save visualization
            debug_path = os.path.splitext(image_path)[0] + "_debug" + os.path.splitext(image_path)[1]
            img.save(debug_path)
            logger.info(f"Saved bounding box visualization to {debug_path}")
            
            return debug_path
        except Exception as e:
            logger.error(f"Error creating bounding box visualization: {str(e)}")
            return None