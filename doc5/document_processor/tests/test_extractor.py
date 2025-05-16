"""
Tests for the text extraction module
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open
import os

from document_processor.core.extraction.text_extractor import (
    BaseExtractor, PdfExtractor, ImageExtractor, DocumentExtractorFactory,
    extract_document_content
)
from tests.test_base import BaseTestCase

class TestBaseExtractor(BaseTestCase):
    """Test cases for BaseExtractor"""
    
    def test_get_file_extension(self):
        """Test file extension extraction"""
        extractor = BaseExtractor()
        
        # Test various file paths
        self.assertEqual(extractor._get_file_extension('/path/to/file.pdf'), '.pdf')
        self.assertEqual(extractor._get_file_extension('document.PDF'), '.pdf')
        self.assertEqual(extractor._get_file_extension('image.jpg'), '.jpg')
        self.assertEqual(extractor._get_file_extension('C:\\Users\\test\\doc.docx'), '.docx')
        self.assertEqual(extractor._get_file_extension('file_without_extension'), '')
    
    def test_extract_not_implemented(self):
        """Test that abstract extract method raises NotImplementedError"""
        extractor = BaseExtractor()
        with self.assertRaises(NotImplementedError):
            extractor.extract('dummy.pdf')

class TestPdfExtractor(BaseTestCase):
    """Test cases for PdfExtractor"""
    
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        
        # Create patches for PyMuPDF and tabula
        self.fitz_patch = patch('document_processor.core.extraction.text_extractor.fitz')
        self.mock_fitz = self.fitz_patch.start()
        
        self.tabula_patch = patch('document_processor.core.extraction.text_extractor.tabula')
        self.mock_tabula = self.tabula_patch.start()
        
        # Set up mock for fitz.open
        self.mock_doc = MagicMock()
        self.mock_page = MagicMock()
        self.mock_page.get_text.return_value = "Sample PDF text"
        self.mock_doc.__iter__.return_value = [self.mock_page]
        self.mock_fitz.open.return_value = self.mock_doc
        
        # Set up mock for tabula.read_pdf
        self.mock_df = MagicMock()
        self.mock_df.to_dict.return_value = [{"col1": "data1", "col2": "data2"}]
        self.mock_tabula.read_pdf.return_value = [self.mock_df]
        
        # Create extractor
        self.pdf_extractor = PdfExtractor()
    
    def tearDown(self):
        """Clean up test environment"""
        self.fitz_patch.stop()
        self.tabula_patch.stop()
        super().tearDown()
    
    def test_extract_pdf(self):
        """Test PDF text extraction"""
        # Extract from sample PDF
        result = self.pdf_extractor.extract(self.sample_pdf_path)
        
        # Verify fitz.open was called
        self.mock_fitz.open.assert_called_with(self.sample_pdf_path)
        
        # Verify get_text was called
        self.mock_page.get_text.assert_called_once()
        
        # Verify tabula.read_pdf was called
        self.mock_tabula.read_pdf.assert_called_with(
            self.sample_pdf_path, pages='all', multiple_tables=True)
        
        # Verify results
        self.assertEqual(result['text'], "Sample PDF text")
        self.assertEqual(result['tables'], [{"col1": "data1", "col2": "data2"}])
    
    def test_extract_invalid_file_type(self):
        """Test extraction with invalid file type"""
        with self.assertRaises(ValueError):
            self.pdf_extractor.extract('file.jpg')
    
    def test_extract_minimal_text(self):
        """Test handling of minimal text extraction (scanned PDF)"""
        # Make get_text return minimal text
        self.mock_page.get_text.return_value = "  "
        
        # Extract from PDF
        result = self.pdf_extractor.extract(self.sample_pdf_path)
        
        # Text should still be returned, but would generate a warning (logged)
        self.assertEqual(result['text'], "  ")

class TestImageExtractor(BaseTestCase):
    """Test cases for ImageExtractor"""
    
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        
        # Create patch for docling
        self.docling_module = MagicMock()
        self.mock_converter = MagicMock()
        self.mock_result = MagicMock()
        self.mock_document = MagicMock()
        
        self.mock_document.export_to_markdown.return_value = "Extracted text from image"
        self.mock_document.tables = []
        self.mock_result.document = self.mock_document
        self.mock_converter.convert.return_value = self.mock_result
        self.docling_module.DocumentConverter.return_value = self.mock_converter
        
        # Apply patch
        self.docling_patch = patch.dict('sys.modules', {'docling.document_converter': self.docling_module})
        self.docling_patch.start()
        
        # Create extractor
        self.image_extractor = ImageExtractor()
    
    def tearDown(self):
        """Clean up test environment"""
        self.docling_patch.stop()
        super().tearDown()
    
    def test_extract_image(self):
        """Test image extraction"""
        # Extract from sample image
        result = self.image_extractor.extract(self.sample_image_path)
        
        # Verify converter was called
        self.mock_converter.convert.assert_called_with(self.sample_image_path)
        
        # Verify markdown export was called
        self.mock_document.export_to_markdown.assert_called_once()
        
        # Verify results
        self.assertEqual(result['text'], "Extracted text from image")
        self.assertEqual(result['tables'], [])
    
    def test_extract_unsupported_file_type(self):
        """Test extraction with unsupported file type"""
        with self.assertRaises(ValueError):
            self.image_extractor.extract('file.pdf')
    
    def test_extract_with_tables(self):
        """Test extraction with tables"""
        # Add tables to mock document
        table1 = MagicMock()
        table1.to_dict.return_value = {"header": ["Name", "Value"], "rows": [["Item1", "100"]]}
        self.mock_document.tables = [table1]
        
        # Extract from image
        result = self.image_extractor.extract(self.sample_image_path)
        
        # Verify tables were extracted
        self.assertEqual(result['tables'], [{"header": ["Name", "Value"], "rows": [["Item1", "100"]]}])

class TestDocumentExtractorFactory(BaseTestCase):
    """Test cases for DocumentExtractorFactory"""
    
    def test_get_extractor_for_pdf(self):
        """Test getting extractor for PDF"""
        extractor = DocumentExtractorFactory.get_extractor('file.pdf')
        self.assertIsInstance(extractor, PdfExtractor)
    
    def test_get_extractor_for_image(self):
        """Test getting extractor for image types"""
        for ext in ['.jpg', '.jpeg', '.png', '.tiff']:
            extractor = DocumentExtractorFactory.get_extractor(f'file{ext}')
            self.assertIsInstance(extractor, ImageExtractor)
    
    def test_get_extractor_unsupported(self):
        """Test getting extractor for unsupported file type"""
        with self.assertRaises(ValueError):
            DocumentExtractorFactory.get_extractor('file.doc')

class TestExtractDocumentContent(BaseTestCase):
    """Test cases for extract_document_content function"""
    
    @patch('document_processor.core.extraction.text_extractor.DocumentExtractorFactory.get_extractor')
    def test_extract_document_content(self, mock_get_extractor):
        """Test extract_document_content function"""
        # Mock extractor
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {"text": "Sample text", "tables": []}
        mock_get_extractor.return_value = mock_extractor
        
        # Call function
        result = extract_document_content('file.pdf')
        
        # Verify extractor was obtained and used
        mock_get_extractor.assert_called_with('file.pdf')
        mock_extractor.extract.assert_called_with('file.pdf')
        
        # Verify result
        self.assertEqual(result, {"text": "Sample text", "tables": []})

if __name__ == '__main__':
    unittest.main()