"""
Tests for the financial information extraction module
"""
import unittest
from unittest.mock import patch, MagicMock

from document_processor.core.information.financial_extractor import FinancialExtractor
from document_processor.db.models import PartnershipDetails, FinancialInformation
from tests.test_base import BaseTestCase

class TestFinancialExtractor(BaseTestCase):
    """Test cases for FinancialExtractor"""
    
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        
        # Create patch for spacy
        self.spacy_patch = patch('document_processor.core.information.financial_extractor.spacy')
        self.mock_spacy = self.spacy_patch.start()
        
        # Mock NLP model
        self.mock_nlp = MagicMock()
        self.mock_spacy.load.return_value = self.mock_nlp
        
        # Mock entity ruler
        self.mock_ruler = MagicMock()
        self.mock_nlp.add_pipe.return_value = self.mock_ruler
        
        # Create extractor
        self.extractor = FinancialExtractor()
    
    def tearDown(self):
        """Clean up test environment"""
        self.spacy_patch.stop()
        super().tearDown()
    
    def test_init(self):
        """Test extractor initialization"""
        # Verify spacy model was loaded
        self.mock_spacy.load.assert_called_once_with("en_core_web_sm")
        
        # Verify entity ruler was added
        self.mock_nlp.add_pipe.assert_called_once_with("entity_ruler")
        
        # Verify patterns were added
        self.mock_ruler.add_patterns.assert_called_once()
    
    def test_extract_partnership_details(self):
        """Test partnership details extraction"""
        # Mock the document entities
        mock_doc = MagicMock()
        org_entity = MagicMock()
        org_entity.text = "ABC Investment Partners"
        org_entity.label_ = "ORG"
        
        person_entity = MagicMock()
        person_entity.text = "John Smith"
        person_entity.label_ = "PERSON"
        
        location_entity = MagicMock()
        location_entity.text = "New York, NY 10001"
        location_entity.label_ = "GPE"
        
        location_entity2 = MagicMock()
        location_entity2.text = "Boston, MA 02110"
        location_entity2.label_ = "GPE"
        
        mock_doc.ents = [org_entity, person_entity, location_entity, location_entity2]
        self.mock_nlp.return_value = mock_doc
        
        # Extract partnership details
        details = self.extractor.extract_partnership_details(self.test_text)
        
        # Verify NLP model was called
        self.mock_nlp.assert_called_once_with(self.test_text)
        
        # Verify extracted details
        self.assertEqual(details.name, "ABC Investment Partners")
        self.assertEqual(details.recipient_name, "John Smith")
        self.assertEqual(details.address, "New York, NY 10001")
        self.assertEqual(details.recipient_address, "Boston, MA 02110")
        self.assertEqual(details.ein, "12-3456789")  # From regex pattern in test_text
    
    def test_extract_section(self):
        """Test section extraction"""
        # Extract financial section
        financial_section = self.extractor.extract_section(
            self.test_text, "NET INCOME", "INSTRUCTIONS")
        
        # Verify extracted section
        self.assertIn("NET INCOME (LOSS): 125,000.00", financial_section)
        self.assertIn("OTHER PORTFOLIO INCOME: 45,500.00", financial_section)
        
        # Test with missing end keyword
        instructions = self.extractor.extract_section(
            self.test_text, "INSTRUCTIONS:", None)
        
        # Verify extracted section goes to end of text
        self.assertIn("File this schedule with your tax return", instructions)
        self.assertIn("Tax Return Due: April 15, 2024", instructions)
        
        # Test with non-existent section
        missing_section = self.extractor.extract_section(
            self.test_text, "NONEXISTENT SECTION", None)
        
        # Verify None is returned
        self.assertIsNone(missing_section)
    
    def test_extract_financial_data(self):
        """Test financial data extraction"""
        # Extract net income
        section = "NET INCOME (LOSS): 125,000.00\nOTHER PORTFOLIO INCOME: 45,500.00"
        
        net_income = self.extractor.extract_financial_data(
            section, "NET INCOME \\(LOSS\\)")
        
        # Verify extracted value
        self.assertEqual(net_income, "125,000.00")
        
        # Test with different format
        section2 = "NET INCOME (LOSS) = $125,000.00\nOTHER PORTFOLIO INCOME: 45,500.00"
        
        net_income2 = self.extractor.extract_financial_data(
            section2, "NET INCOME \\(LOSS\\)")
        
        # Verify extracted value
        self.assertEqual(net_income2, "125,000.00")
        
        # Test with None section
        none_result = self.extractor.extract_financial_data(None, "NET INCOME")
        
        # Verify None is returned
        self.assertIsNone(none_result)
        
        # Test with non-existent data
        missing_data = self.extractor.extract_financial_data(
            section, "NONEXISTENT DATA")
        
        # Verify None is returned
        self.assertIsNone(missing_data)
    
    def test_extract_financial_information(self):
        """Test financial information extraction"""
        # Extract financial information
        financial_info, tax_instructions, tax_summary = (
            self.extractor.extract_financial_information(self.test_text))
        
        # Verify financial information
        self.assertIsInstance(financial_info, FinancialInformation)
        self.assertEqual(financial_info.net_income, "125,000.00")
        self.assertEqual(financial_info.portfolio_income, "45,500.00")
        
        # Verify tax instructions
        self.assertIn("File this schedule with your tax return", tax_instructions)
        self.assertIn("Please consult your tax advisor", tax_instructions)
        
        # Verify tax summary (should be same as instructions in this case)
        self.assertEqual(tax_summary, tax_instructions)
        
        # Test with very long tax instructions
        long_instructions = "INSTRUCTIONS: " + "Sample tax instruction. " * 200
        text_with_long_instructions = (
            self.test_text + long_instructions)
        
        _, _, summary = self.extractor.extract_financial_information(
            text_with_long_instructions)
        
        # Verify summary is truncated
        self.assertIn("...", summary)
        self.assertLess(len(summary.split()), 105)  # 100 words plus "..."

if __name__ == '__main__':
    unittest.main()