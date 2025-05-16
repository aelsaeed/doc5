"""
Document classification module using Sentence-BERT
"""
import logging
from typing import Dict, Tuple, List
import torch
from sentence_transformers import SentenceTransformer, util

from document_processor.utils.gpu_utils import check_gpu_availability

logger = logging.getLogger(__name__)

class DocumentClassifier:
    """Document type classifier using Sentence-BERT for semantic similarity"""
    
    # Document types and descriptions
    DOCUMENT_TYPES = [
        "K1 (Schedule K-1)", "K2 (Schedule K-2)", "W1 (Form W-1)", "W2 (Form W-2)", "Tax Return",
        "Shareholder Meeting Notice", "Proxy Statement", "Financial Statement",
        "SEC Filing", "Loan Agreement", "Investment Agreement"
    ]
    
    TYPE_DESCRIPTIONS = {
        "K1 (Schedule K-1)": "A Schedule K-1 (Form 1065) document detailing a partner's share of income, deductions, credits, and other financial items in a partnership.",
        "K2 (Schedule K-2)": "A Schedule K-2 document providing supplemental information for international tax reporting related to a partnership.",
        "W1 (Form W-1)": "A Form W-1 document reporting wage and tax information for employees.",
        "W2 (Form W-2)": "A Form W-2 wage and tax statement document reporting an employee's annual wages and taxes withheld.",
        "Tax Return": "A tax return document, such as Form 1040, reporting an individual's income and tax obligations.",
        "Shareholder Meeting Notice": "A notice announcing an upcoming shareholder meeting, often including voting information.",
        "Proxy Statement": "A proxy statement providing information for shareholders to vote on company matters.",
        "Financial Statement": "A financial statement including balance sheets, income statements, or cash flow statements.",
        "SEC Filing": "An SEC filing, such as Form 10-K or 10-Q, reporting financial information to the SEC.",
        "Loan Agreement": "A loan agreement outlining terms between a borrower and lender.",
        "Investment Agreement": "An investment agreement detailing the terms of an investment between parties."
    }
    
    TYPE_KEYWORDS = {
        "K1 (Schedule K-1)": ["schedule k-1", "form 1065", "partner's share", "partnership", "tax year"],
        "K2 (Schedule K-2)": ["schedule k-2", "supplemental information", "international tax"],
        "W1 (Form W-1)": ["form w-1", "wage and tax statement"],
        "W2 (Form W-2)": ["w-2", "wage and tax statement", "form w-2", "employee's federal tax return", 
                         "employer identification number", "social security wages"],
        "Tax Return": ["tax return", "form 1040", "income tax"],
        "Shareholder Meeting Notice": ["shareholder meeting", "annual meeting", "proxy statement"],
        "Proxy Statement": ["proxy statement", "voting", "board of directors"],
        "Financial Statement": ["financial statement", "balance sheet", "income statement", "cash flow"],
        "SEC Filing": ["sec filing", "form 10-k", "form 10-q", "edgar"],
        "Loan Agreement": ["loan agreement", "borrower", "lender", "terms and conditions"],
        "Investment Agreement": ["investment agreement", "investor", "equity", "terms of investment"]
    }
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the document classifier
        
        Args:
            model_name (str): Name of the Sentence-BERT model to use
        """
        self.device = check_gpu_availability()
        self.model = SentenceTransformer(model_name, device=self.device)
        # Cache embeddings for type descriptions
        self.description_embeddings = self._get_description_embeddings()
        logger.info(f"Document classifier initialized with model: {model_name}")
    
    def _get_description_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Generate embeddings for all document type descriptions
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping document types to their embeddings
        """
        return {doc_type: self.model.encode(desc) 
                for doc_type, desc in self.TYPE_DESCRIPTIONS.items()}
    
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify document type using semantic similarity
        
        Args:
            text (str): Document text to classify
            
        Returns:
            Tuple[str, float]: Tuple containing document type and confidence score
        """
        # Check for W-2 form explicitly based on keywords
        text_lower = text.lower()
        w2_keywords = [
            "w-2", "wage and tax statement", "form w-2",
            "employee's social security number", "employer identification number"
        ]
        
        if any(keyword in text_lower for keyword in w2_keywords) and "copy b" in text_lower:
            logger.info("Document classified as W2 (Form W-2) based on keywords")
            return "W2 (Form W-2)", 0.95
        
        # Continue with standard classification if no W-2 specific match
        # Limit to 512 characters to avoid memory issues with large documents
        text_sample = text[:512]
        
        # Generate embedding for the input text
        text_embedding = self.model.encode(text_sample)
        
        # Calculate similarity with all document type descriptions
        similarities = {doc_type: util.pytorch_cos_sim(text_embedding, emb).item() 
                      for doc_type, emb in self.description_embeddings.items()}
        
        # Find the document type with highest similarity
        best_match = max(similarities, key=similarities.get)
        confidence = similarities[best_match]
        
        logger.info(f"Document classified as '{best_match}' with confidence {confidence:.2f}")
        return best_match, confidence
    
    def generate_reasoning(self, text: str, doc_type: str) -> str:
        """
        Generate reasoning for the classification decision
        
        Args:
            text (str): Document text
            doc_type (str): Classified document type
            
        Returns:
            str: Reasoning for the classification
        """
        text_lower = text.lower()
        keywords = self.TYPE_KEYWORDS.get(doc_type, [])
        found_keywords = [kw for kw in keywords if kw in text_lower]
        
        if found_keywords:
            return f"The document contains keywords such as {', '.join(found_keywords)}, which are typical for {doc_type} documents."
        return f"The document was classified as {doc_type} based on its content."
    
    def generate_summary(self, doc_type: str, text: str) -> str:
        """
        Generate a human-readable summary for the document
        
        Args:
            doc_type (str): Document type
            text (str): Document text
            
        Returns:
            str: Summary text
        """
        summaries = {
            "K1 (Schedule K-1)": "The document provided is a Schedule K-1 (Form 1065). It includes detailed information about the partner's share of income, deductions, credits, and other financial items related to the partnership. The document also contains a letter explaining the purpose of the Schedule K-1 and instructions for the recipient. This matches the characteristics of a financial K-1 document, typically prepared for partners in a partnership.",
            "K2 (Schedule K-2)": "The document provided is a Schedule K-2, which includes supplemental information for international tax reporting related to a partnership. This matches the characteristics of a financial K-2 document, typically prepared for partners with international tax obligations.",
            "W1 (Form W-1)": "The document provided is a Form W-1, which reports wage and tax information. This matches the characteristics of a W-1 document, typically prepared for employees.",
            "W2 (Form W-2)": "The document provided is a Form W-2 Wage and Tax Statement. It includes tax information such as the employee's wages, tips, and other compensation, as well as federal, state, and local taxes withheld. This document is prepared by employers for employees and is used for filing tax returns.",
            "Tax Return": "The document provided is a Tax Return, likely a Form 1040, which reports an individual's income and tax obligations. This matches the characteristics of a tax return document.",
            "Shareholder Meeting Notice": "The document provided is a Shareholder Meeting Notice, which announces an upcoming meeting for shareholders. This matches the characteristics of a shareholder meeting notice.",
            "Proxy Statement": "The document provided is a Proxy Statement, which provides information for shareholders to vote on company matters. This matches the characteristics of a proxy statement.",
            "Financial Statement": "The document provided is a Financial Statement, which includes financial data such as balance sheets or income statements. This matches the characteristics of a financial statement.",
            "SEC Filing": "The document provided is an SEC Filing, such as a Form 10-K or 10-Q, which reports financial information to the SEC. This matches the characteristics of an SEC filing.",
            "Loan Agreement": "The document provided is a Loan Agreement, which outlines the terms between a borrower and lender. This matches the characteristics of a loan agreement.",
            "Investment Agreement": "The document provided is an Investment Agreement, which details the terms of an investment between parties. This matches the characteristics of an investment agreement."
        }
        
        return summaries.get(doc_type, f"The document is classified as {doc_type} based on its content.")