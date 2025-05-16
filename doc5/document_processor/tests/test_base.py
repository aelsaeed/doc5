"""
Base test class for document processor tests
"""
import os
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from document_processor.main import create_app
from document_processor.config import TestingConfig

class BaseTestCase(unittest.TestCase):
    """Base test class with common setup and teardown"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for uploads
        self.temp_dir = tempfile.mkdtemp()
        
        # Override upload folder config
        TestingConfig.UPLOAD_FOLDER = self.temp_dir
        
        # Create app with testing config
        self.app = create_app(config_class=TestingConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
        
        # Create test client
        self.client = self.app.test_client()
        
        # Sample test data
        self.test_text = """
        SCHEDULE K-1 PARTNERSHIP
        Partner's Share of Income, Deductions, Credits, etc.
        
        Partnership: ABC Investment Partners
        123 Finance Street
        New York, NY 10001
        EIN: 12-3456789
        
        Partner: John Smith
        456 Investor Avenue
        Boston, MA 02110
        
        NET INCOME (LOSS): 125,000.00
        OTHER PORTFOLIO INCOME: 45,500.00
        
        INSTRUCTIONS:
        File this schedule with your tax return. Keep a copy for your records.
        This K-1 is for tax year 2023. Please consult your tax advisor for
        proper reporting of these items on your tax return.
        
        Tax Return Due: April 15, 2024
        Dividend Payment Date: February 28, 2024
        """
        
        # Create sample PDF for testing
        self.create_sample_files()
    
    def tearDown(self):
        """Clean up test environment"""
        # Pop app context
        self.app_context.pop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def create_sample_files(self):
        """Create sample files for testing"""
        # For testing, we'll just create empty files with proper extensions
        self.sample_pdf_path = os.path.join(self.temp_dir, 'sample.pdf')
        with open(self.sample_pdf_path, 'w') as f:
            f.write("Sample PDF content")
        
        self.sample_image_path = os.path.join(self.temp_dir, 'sample.jpg')
        with open(self.sample_image_path, 'w') as f:
            f.write("Sample image content")
    
    def get_mock_pdf_content(self):
        """Get mock content for PDF testing"""
        return {
            "text": self.test_text,
            "tables": [
                [
                    {"Account": "Investment A", "Value": "100000"},
                    {"Account": "Investment B", "Value": "250000"}
                ]
            ]
        }