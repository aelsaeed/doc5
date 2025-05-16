"""
Integration tests for API endpoints
"""
import os
import json
import unittest
from unittest.mock import patch, MagicMock
import io
import tempfile
from pathlib import Path

from document_processor.db.models import generate_session_id
from tests.test_base import BaseTestCase

class TestApiEndpoints(BaseTestCase):
    """Integration tests for API endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        
        # Create sample files with different extensions
        self.valid_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', dir=self.temp_dir, delete=False)
        self.valid_pdf.write(b"Sample PDF content")
        self.valid_pdf.close()
        
        self.valid_jpg = tempfile.NamedTemporaryFile(suffix='.jpg', dir=self.temp_dir, delete=False)
        self.valid_jpg.write(b"Sample JPG content")
        self.valid_jpg.close()
        
        self.invalid_file = tempfile.NamedTemporaryFile(suffix='.docx', dir=self.temp_dir, delete=False)
        self.invalid_file.write(b"Sample DOCX content")
        self.invalid_file.close()
        
        # Create sample session ID
        self.test_session_id = generate_session_id()
        
        # Register patches
        self.processor_patch = patch('document_processor.api.routes.processor')
        self.mock_processor = self.processor_patch.start()
        
        self.get_results_patch = patch('document_processor.api.routes.get_processing_results')
        self.mock_get_results = self.get_results_patch.start()
        
        self.save_correction_patch = patch('document_processor.api.routes.save_correction')
        self.mock_save_correction = self.save_correction_patch.start()
        
        self.get_investments_patch = patch('document_processor.api.routes.get_investments')
        self.mock_get_investments = self.get_investments_patch.start()
    
    def tearDown(self):
        """Clean up test environment"""
        # Stop patches
        self.processor_patch.stop()
        self.get_results_patch.stop()
        self.save_correction_patch.stop()
        self.get_investments_patch.stop()
        
        # Clean up temporary files
        for filepath in [self.valid_pdf.name, self.valid_jpg.name, self.invalid_file.name]:
            if os.path.exists(filepath):
                os.unlink(filepath)
                
        super().tearDown()
    
    def test_upload_valid_pdf(self):
        """Test uploading a valid PDF file"""
        # Mock processor to return a session ID
        self.mock_processor.process_document.return_value = self.test_session_id
        
        # Create test file
        with open(self.valid_pdf.name, 'rb') as f:
            data = {'file': (f, Path(self.valid_pdf.name).name)}
            response = self.client.post('/api/upload', 
                                        data=data, 
                                        content_type='multipart/form-data')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertTrue(json_data['success'])
        self.assertEqual(json_data['session_id'], self.test_session_id)
        
        # Verify processor was called
        self.mock_processor.process_document.assert_called_once()
    
    def test_upload_valid_jpg(self):
        """Test uploading a valid JPG file"""
        # Mock processor to return a session ID
        self.mock_processor.process_document.return_value = self.test_session_id
        
        # Create test file
        with open(self.valid_jpg.name, 'rb') as f:
            data = {'file': (f, Path(self.valid_jpg.name).name)}
            response = self.client.post('/api/upload', 
                                        data=data, 
                                        content_type='multipart/form-data')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertTrue(json_data['success'])
        self.assertEqual(json_data['session_id'], self.test_session_id)
        
        # Verify processor was called
        self.mock_processor.process_document.assert_called_once()
    
    def test_upload_invalid_file_type(self):
        """Test uploading an invalid file type"""
        # Set up processor to raise FileTypeError
        from document_processor.utils.custom_exceptions import FileTypeError
        self.mock_processor.process_document.side_effect = FileTypeError(
            self.invalid_file.name, ['.pdf', '.jpg', '.jpeg', '.png', '.tiff'])
        
        # Create test file
        with open(self.invalid_file.name, 'rb') as f:
            data = {'file': (f, Path(self.invalid_file.name).name)}
            response = self.client.post('/api/upload', 
                                        data=data, 
                                        content_type='multipart/form-data')
        
        # Verify response
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['error'], 'FileTypeError')
        self.assertIn('Unsupported file type', json_data['message'])
    
    def test_upload_no_file(self):
        """Test uploading with no file"""
        # No file in request
        response = self.client.post('/api/upload')
        
        # Verify response
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['error'], 'No file uploaded')
    
    def test_upload_empty_filename(self):
        """Test uploading with empty filename"""
        # Empty filename
        data = {'file': (io.BytesIO(b"test data"), '')}
        response = self.client.post('/api/upload', 
                                    data=data, 
                                    content_type='multipart/form-data')
        
        # Verify response
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['error'], 'No file selected')
    
    def test_status_processing(self):
        """Test status endpoint when document is still processing"""
        # Mock get_processing_results to return None (still processing)
        self.mock_get_results.return_value = None
        
        # Get status
        response = self.client.get(f'/api/status/{self.test_session_id}')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['status'], 'processing')
    
    def test_status_completed(self):
        """Test status endpoint when processing is complete"""
        # Mock get_processing_results to return results
        self.mock_get_results.return_value = {"summary": {"document_type": "K1"}}
        
        # Get status
        response = self.client.get(f'/api/status/{self.test_session_id}')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['status'], 'completed')
    
    def test_status_error(self):
        """Test status endpoint when processing had an error"""
        # Mock get_processing_results to return error
        self.mock_get_results.return_value = {"error": "Something went wrong"}
        
        # Get status
        response = self.client.get(f'/api/status/{self.test_session_id}')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['status'], 'error')
        self.assertEqual(json_data['message'], "Something went wrong")
    
    def test_status_invalid_session_id(self):
        """Test status endpoint with invalid session ID"""
        # Invalid session ID
        response = self.client.get('/api/status/invalid-id')
        
        # Verify response
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['error'], 'ValidationError')
    
    def test_results_not_found(self):
        """Test results endpoint when results are not found"""
        # Mock get_processing_results to return None
        self.mock_get_results.return_value = None
        
        # Get results
        response = self.client.get(f'/api/results/{self.test_session_id}')
        
        # Verify response
        self.assertEqual(response.status_code, 404)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['error'], 'NotFound')
    
    def test_results_with_error(self):
        """Test results endpoint when processing had an error"""
        # Mock get_processing_results to return error
        self.mock_get_results.return_value = {"error": "Processing failed"}
        
        # Get results
        response = self.client.get(f'/api/results/{self.test_session_id}')
        
        # Verify response
        self.assertEqual(response.status_code, 500)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['error'], 'ProcessingError')
        self.assertEqual(json_data['message'], "Processing failed")
    
    def test_results_success(self):
        """Test results endpoint with successful processing"""
        # Mock get_processing_results to return results
        mock_results = {
            "summary": {"document_type": "K1", "confidence": 0.95},
            "text": "Sample text",
            "tables": [],
            "investment": {"id": 1, "name": "Investment A"}
        }
        self.mock_get_results.return_value = mock_results
        
        # Get results
        response = self.client.get(f'/api/results/{self.test_session_id}')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertEqual(json_data, mock_results)
    
    def test_results_simplified(self):
        """Test results endpoint with simplified option"""
        # Mock get_processing_results to return results
        mock_results = {
            "summary": {"document_type": "K1", "confidence": 0.95},
            "text": "Sample text",
            "tables": [],
            "investment": {"id": 1, "name": "Investment A"}
        }
        self.mock_get_results.return_value = mock_results
        
        # Get simplified results
        response = self.client.get(f'/api/results/{self.test_session_id}?simplified=true')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['summary'], mock_results['summary'])
        self.assertEqual(json_data['investment'], mock_results['investment'])
        self.assertNotIn('text', json_data)
        self.assertNotIn('tables', json_data)
    
    def test_correct_success(self):
        """Test correction endpoint with valid data"""
        # Mock save_correction to return True
        self.mock_save_correction.return_value = True
        
        # Submit correction
        corrections = {"correct_type": "Tax Return"}
        response = self.client.post(
            f'/api/correct/{self.test_session_id}',
            json=corrections,
            content_type='application/json'
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertTrue(json_data['success'])
        
        # Verify save_correction was called
        self.mock_save_correction.assert_called_once_with(self.test_session_id, corrections)
    
    def test_correct_not_json(self):
        """Test correction endpoint with non-JSON data"""
        # Submit non-JSON data
        response = self.client.post(
            f'/api/correct/{self.test_session_id}',
            data="not json",
            content_type='text/plain'
        )
        
        # Verify response
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['error'], 'InvalidRequest')
    
    def test_correct_missing_required_field(self):
        """Test correction endpoint with missing required field"""
        # Submit data without required field
        response = self.client.post(
            f'/api/correct/{self.test_session_id}',
            json={},
            content_type='application/json'
        )
        
        # Verify response
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['error'], 'ValidationError')
    
    def test_investments_list(self):
        """Test investments endpoint"""
        # Mock get_investments to return test data
        mock_investments = [
            {"id": 1, "name": "Investment A"},
            {"id": 2, "name": "Investment B"}
        ]
        self.mock_get_investments.return_value = mock_investments
        
        # Get investments
        response = self.client.get('/api/investments')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['investments'], mock_investments)
    
    def test_document_types(self):
        """Test document_types endpoint"""
        # Get document types
        response = self.client.get('/api/document_types')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertIn('document_types', json_data)
        self.assertIn('K1 (Schedule K-1)', json_data['document_types'])
        self.assertIn('Tax Return', json_data['document_types'])
    
    def test_health_check(self):
        """Test health check endpoint"""
        # Get health status
        response = self.client.get('/api/health')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['status'], 'ok')
    
    def test_reprocess_success(self):
        """Test reprocess endpoint with successful reprocessing"""
        # Mock reprocess_document to return True
        self.mock_processor.reprocess_document.return_value = True
        
        # Reprocess document
        response = self.client.post(f'/api/reprocess/{self.test_session_id}')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertTrue(json_data['success'])
        
        # Verify reprocess_document was called
        self.mock_processor.reprocess_document.assert_called_once_with(self.test_session_id)
    
    def test_reprocess_failure(self):
        """Test reprocess endpoint when reprocessing fails"""
        # Mock reprocess_document to return False
        self.mock_processor.reprocess_document.return_value = False
        
        # Reprocess document
        response = self.client.post(f'/api/reprocess/{self.test_session_id}')
        
        # Verify response
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['error'], 'ReprocessingError')
    
    def test_reprocess_file_not_found(self):
        """Test reprocess endpoint when file is not found"""
        # Mock reprocess_document to raise FileReadError
        from document_processor.utils.custom_exceptions import FileReadError
        self.mock_processor.reprocess_document.side_effect = FileReadError(
            "test-file.pdf", "File no longer exists")
        
        # Reprocess document
        response = self.client.post(f'/api/reprocess/{self.test_session_id}')
        
        # Verify response
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertEqual(json_data['error'], 'FileReadError')
        self.assertIn('File no longer exists', json_data['message'])

if __name__ == '__main__':
    unittest.main()