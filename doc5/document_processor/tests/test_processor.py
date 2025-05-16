"""
Tests for the document processor service
"""
import unittest
from unittest.mock import patch, MagicMock, call

from document_processor.core.processor_service import DocumentProcessor
from document_processor.db.models import (
    ProcessingResult, DocumentSummary, PartnershipDetails, 
    FinancialInformation, Entity, ImportantDate, Investment
)
from tests.test_base import BaseTestCase

class TestDocumentProcessor(BaseTestCase):
    """Test cases for DocumentProcessor"""
    
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        
        # Create patches
        self.extract_patch = patch('document_processor.core.processor_service.extract_document_content')
        self.mock_extract = self.extract_patch.start()
        
        self.classifier_patch = patch('document_processor.core.processor_service.DocumentClassifier')
        self.mock_classifier_class = self.classifier_patch.start()
        self.mock_classifier = MagicMock()
        self.mock_classifier_class.return_value = self.mock_classifier
        
        self.financial_patch = patch('document_processor.core.processor_service.FinancialExtractor')
        self.mock_financial_class = self.financial_patch.start()
        self.mock_financial = MagicMock()
        self.mock_financial_class.return_value = self.mock_financial
        
        self.date_patch = patch('document_processor.core.processor_service.DateExtractor')
        self.mock_date_class = self.date_patch.start()
        self.mock_date = MagicMock()
        self.mock_date_class.return_value = self.mock_date
        
        self.save_patch = patch('document_processor.core.processor_service.save_processing_results')
        self.mock_save = self.save_patch.start()
        
        self.find_patch = patch('document_processor.core.processor_service.find_investment_by_name')
        self.mock_find = self.find_patch.start()
        
        self.mp_patch = patch('document_processor.core.processor_service.multiprocessing')
        self.mock_mp = self.mp_patch.start()
        
        # Mock process object
        self.mock_process = MagicMock()
        self.mock_mp.Process.return_value = self.mock_process
        
        # Set up test data
        self.mock_extract.return_value = {
            "text": self.test_text,
            "tables": []
        }
        
        self.mock_classifier.classify.return_value = ("K1 (Schedule K-1)", 0.95)
        self.mock_classifier.generate_reasoning.return_value = "Sample reasoning"
        self.mock_classifier.generate_summary.return_value = "Sample summary"
        
        self.mock_financial.extract_partnership_details.return_value = PartnershipDetails(
            name="ABC Investment Partners",
            address="New York, NY 10001",
            ein="12-3456789",
            recipient_name="John Smith",
            recipient_address="Boston, MA 02110"
        )
        
        self.mock_financial.extract_financial_information.return_value = (
            FinancialInformation(
                net_income="125,000.00",
                portfolio_income="45,500.00"
            ),
            "Sample tax instructions",
            "Sample tax summary"
        )
        
        self.mock_date.extract_important_dates.return_value = [
            ImportantDate(event="Tax Return Submission", date="2024-04-15"),
            ImportantDate(event="Dividend Payment", date="2024-02-28")
        ]
        
        self.mock_find.return_value = {"id": 1, "name": "Investment A"}
        
        # Create processor
        self.processor = DocumentProcessor()
    
    def tearDown(self):
        """Clean up test environment"""
        self.extract_patch.stop()
        self.classifier_patch.stop()
        self.financial_patch.stop()
        self.date_patch.stop()
        self.save_patch.stop()
        self.find_patch.stop()
        self.mp_patch.stop()
        super().tearDown()
    
    def test_process_document(self):
        """Test document processing"""
        # Process document
        session_id = self.processor.process_document(self.sample_pdf_path)
        
        # Verify session ID was saved
        self.mock_save.assert_called_with(session_id, file_path=self.sample_pdf_path)
        
        # Verify process was created and started
        self.mock_mp.Process.assert_called_once()
        self.assertEqual(self.mock_mp.Process.call_args[1]['target'], 
                         self.processor._process_document_async)
        self.mock_process.start.assert_called_once()
    
    @patch('document_processor.core.processor_service.generate_session_id')
    def test_process_document_async(self, mock_generate_id):
        """Test asynchronous document processing"""
        # Set up mock session ID
        mock_session_id = "test-session-123"
        
        # Call private method directly
        self.processor._process_document_async(self.sample_pdf_path, mock_session_id)
        
        # Verify content extraction
        self.mock_extract.assert_called_with(self.sample_pdf_path)
        
        # Verify classification
        self.mock_classifier.classify.assert_called_with(self.test_text)
        self.mock_classifier.generate_reasoning.assert_called_with(
            self.test_text, "K1 (Schedule K-1)")
        self.mock_classifier.generate_summary.assert_called_with(
            "K1 (Schedule K-1)", self.test_text)
        
        # Verify financial extraction
        self.mock_financial.extract_partnership_details.assert_called_with(self.test_text)
        self.mock_financial.extract_financial_information.assert_called_with(self.test_text)
        
        # Verify date extraction
        self.mock_date.extract_important_dates.assert_called_with(self.test_text)
        
        # Verify investment lookup
        self.mock_find.assert_called_with("ABC Investment Partners")
        
        # Verify results saved
        self.mock_save.assert_called_with(mock_session_id, results=unittest.mock.ANY)
        
        # Verify structure of saved results
        saved_results = self.mock_save.call_args[1]['results']
        
        # Check top-level structure
        self.assertIn('summary', saved_results)
        self.assertIn('text', saved_results)
        self.assertIn('entities', saved_results)
        self.assertIn('tables', saved_results)
        self.assertIn('important_dates', saved_results)
        self.assertIn('investment', saved_results)
        self.assertIn('file_path', saved_results)
        
        # Check summary structure
        summary = saved_results['summary']
        self.assertEqual(summary['document_type'], "K1 (Schedule K-1)")
        self.assertEqual(summary['confidence'], 0.95)
        self.assertEqual(summary['summary_text'], "Sample summary")
        self.assertEqual(summary['reasoning'], "Sample reasoning")
        
        # Check partnership details
        partnership = summary['partnership_details']
        self.assertEqual(partnership['name'], "ABC Investment Partners")
        self.assertEqual(partnership['ein'], "12-3456789")
        
        # Check financial information
        financial = summary['financial_information']
        self.assertEqual(financial['net_income'], "125,000.00")
        self.assertEqual(financial['portfolio_income'], "45,500.00")
        
        # Check investment data
        investment = saved_results['investment']
        self.assertEqual(investment['id'], 1)
        self.assertEqual(investment['name'], "Investment A")
    
    def test_process_document_async_with_error(self):
        """Test error handling in asynchronous processing"""
        # Make extract_document_content raise an exception
        self.mock_extract.side_effect = Exception("Test extraction error")
        
        # Call private method
        self.processor._process_document_async(self.sample_pdf_path, "test-session")
        
        # Verify error was saved
        self.mock_save.assert_called_with(
            "test-session", 
            results={"error": "Processing failed: Test extraction error"}
        )
    
    def test_process_document_async_empty_text(self):
        """Test handling of empty extracted text"""
        # Make extract_document_content return empty text
        self.mock_extract.return_value = {
            "text": "   ",
            "tables": []
        }
        
        # Call private method
        self.processor._process_document_async(self.sample_pdf_path, "test-session")
        
        # Verify error was saved
        self.mock_save.assert_called_with(
            "test-session", 
            results={"error": "No text extracted from document"}
        )

if __name__ == '__main__':
    unittest.main()