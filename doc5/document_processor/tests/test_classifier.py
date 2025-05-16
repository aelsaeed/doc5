"""
Tests for the document classifier module
"""
import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np

from document_processor.core.classification.classifier import DocumentClassifier
from tests.test_base import BaseTestCase

class TestDocumentClassifier(BaseTestCase):
    """Test cases for DocumentClassifier"""
    
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        
        # Create patches for SentenceTransformer
        self.model_patch = patch('document_processor.core.classification.classifier.SentenceTransformer')
        self.mock_model_class = self.model_patch.start()
        self.mock_model = MagicMock()
        self.mock_model_class.return_value = self.mock_model
        
        # Create patches for util.pytorch_cos_sim
        self.cos_sim_patch = patch('document_processor.core.classification.classifier.util.pytorch_cos_sim')
        self.mock_cos_sim = self.cos_sim_patch.start()
        
        # Create document classifier
        self.classifier = DocumentClassifier()
    
    def tearDown(self):
        """Clean up test environment"""
        self.model_patch.stop()
        self.cos_sim_patch.stop()
        super().tearDown()
    
    def test_init(self):
        """Test classifier initialization"""
        # Verify model was initialized
        self.mock_model_class.assert_called_once_with('all-MiniLM-L6-v2', device=self.classifier.device)
        
        # Verify description embeddings were created
        self.assertEqual(len(self.classifier.description_embeddings), 
                         len(self.classifier.TYPE_DESCRIPTIONS))
    
    def test_classify_k1_document(self):
        """Test classification of K1 document"""
        # Mock the embedding for sample text
        sample_embedding = torch.tensor([0.1, 0.2, 0.3])
        self.mock_model.encode.return_value = sample_embedding
        
        # Mock the cosine similarity to return highest value for K1
        def mock_cos_sim(emb1, emb2):
            if emb2 is self.classifier.description_embeddings["K1 (Schedule K-1)"]:
                return torch.tensor([[0.95]])
            else:
                return torch.tensor([[0.3]])
        
        self.mock_cos_sim.side_effect = mock_cos_sim
        
        # Test classification
        doc_type, confidence = self.classifier.classify(self.test_text)
        
        # Verify correct classification
        self.assertEqual(doc_type, "K1 (Schedule K-1)")
        self.assertEqual(confidence, 0.95)
        
        # Verify model was called with text sample (limited to 512 chars)
        self.mock_model.encode.assert_called_with(self.test_text[:512])
    
    def test_classify_tax_return_document(self):
        """Test classification of Tax Return document"""
        # Sample tax return text
        tax_return_text = """
        Form 1040 U.S. Individual Income Tax Return
        For the year Jan. 1 - Dec. 31, 2023
        
        Your first name and initial: John M
        Last name: Smith
        Social security number: 123-45-6789
        """
        
        # Mock the embedding for sample text
        sample_embedding = torch.tensor([0.1, 0.2, 0.3])
        self.mock_model.encode.return_value = sample_embedding
        
        # Mock the cosine similarity to return highest value for Tax Return
        def mock_cos_sim(emb1, emb2):
            if emb2 is self.classifier.description_embeddings["Tax Return"]:
                return torch.tensor([[0.88]])
            else:
                return torch.tensor([[0.2]])
        
        self.mock_cos_sim.side_effect = mock_cos_sim
        
        # Test classification
        doc_type, confidence = self.classifier.classify(tax_return_text)
        
        # Verify correct classification
        self.assertEqual(doc_type, "Tax Return")
        self.assertEqual(confidence, 0.88)
    
    def test_generate_reasoning_with_keywords(self):
        """Test reasoning generation with keywords"""
        # Add keywords to text
        text_with_keywords = "This is a schedule k-1 form 1065 for a partnership with tax year 2023."
        
        # Get reasoning
        reasoning = self.classifier.generate_reasoning(text_with_keywords, "K1 (Schedule K-1)")
        
        # Verify reasoning contains keywords
        self.assertIn("schedule k-1", reasoning)
        self.assertIn("form 1065", reasoning)
        self.assertIn("partnership", reasoning)
        self.assertIn("tax year", reasoning)
    
    def test_generate_reasoning_without_keywords(self):
        """Test reasoning generation without keywords"""
        # Text without keywords
        text_without_keywords = "This is a financial document."
        
        # Get reasoning
        reasoning = self.classifier.generate_reasoning(
            text_without_keywords, "K1 (Schedule K-1)")
        
        # Verify generic reasoning
        self.assertIn("The document was classified as K1 (Schedule K-1) based on its content", 
                      reasoning)
    
    def test_generate_summary(self):
        """Test summary generation"""
        # Get summary for K1
        summary = self.classifier.generate_summary("K1 (Schedule K-1)", self.test_text)
        
        # Verify summary content
        self.assertIn("Schedule K-1 (Form 1065)", summary)
        self.assertIn("partner's share of income", summary)
        
        # Test with unknown document type
        unknown_summary = self.classifier.generate_summary("Unknown Type", self.test_text)
        self.assertIn("The document is classified as Unknown Type", unknown_summary)

if __name__ == '__main__':
    unittest.main()