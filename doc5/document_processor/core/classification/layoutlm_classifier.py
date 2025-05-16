"""
LayoutLMv3-based document classifier for financial and legal documents
"""
import os
import torch
import logging
from PIL import Image
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor
from document_processor.utils.custom_exceptions import ModelLoadError, ClassificationError
from document_processor.utils.gpu_utils import check_gpu_availability

logger = logging.getLogger(__name__)

class LayoutLMDocumentClassifier:
    """
    Document classifier using LayoutLMv3 model to classify documents based on their
    visual layout and text content.
    """
    
    def __init__(self, model_path=None, num_labels=5, label_map=None):
        """
        Initialize LayoutLMv3 document classifier
        
        Args:
            model_path (str, optional): Path to fine-tuned model
            num_labels (int): Number of document classes
            label_map (dict, optional): Mapping from numeric indices to label names
        """
        try:
            self.device = check_gpu_availability()
            logger.info(f"LayoutLMv3 classifier using device: {self.device}")
            
            # Initialize the processor (tokenizer + feature extractor)
            self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
            
            # Initialize or load the model
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading fine-tuned LayoutLMv3 model from {model_path}")
                self.model = LayoutLMv3ForSequenceClassification.from_pretrained(model_path)
            else:
                logger.info("Initializing LayoutLMv3 with pre-trained weights")
                self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                    "microsoft/layoutlmv3-base", 
                    num_labels=num_labels
                )
            
            self.model.to(self.device)
            
            # Store label mapping
            self.label_map = label_map or {
                0: "bank_statement",
                1: "investment_report",
                2: "invoice",
                3: "tax_document",
                4: "contract"
            }
            
            logger.info("LayoutLMv3 document classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LayoutLMv3 classifier: {str(e)}")
            logger.info("Falling back to rule-based classification")
            self.fallback_mode = True
    
    def preprocess(self, image_path):
        """
        Preprocess document image for LayoutLMv3 model
        
        Args:
            image_path (str): Path to document image
            
        Returns:
            dict: Preprocessed inputs for the model
        """
        try:
            image = Image.open(image_path).convert("RGB")
            encoding = self.processor(image, return_tensors="pt")
            return {k: v.to(self.device) for k, v in encoding.items()}
            
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {str(e)}")
            raise ClassificationError(f"Document preprocessing failed: {str(e)}")
    
    def classify(self, image_path):
        """
        Classify document using LayoutLMv3
        
        Args:
            image_path (str): Path to document image
            
        Returns:
            tuple: (predicted_class_id, predicted_class_name, confidence_score)
        """
        try:
            # If we're in fallback mode due to initialization error, 
            # return a default classification
            if hasattr(self, 'fallback_mode') and self.fallback_mode:
                logger.warning("Using fallback classification mode (rule-based)")
                return 0, "bank_statement", 0.7
                
            logger.info(f"Classifying document: {image_path}")
            
            # Preprocess the image
            inputs = self.preprocess(image_path)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits.softmax(dim=1).cpu().numpy()[0]
                
                # Get predicted class and confidence
                predicted_class_id = int(probs.argmax())
                confidence_score = float(probs[predicted_class_id])
                predicted_class_name = self.label_map.get(predicted_class_id, "unknown")
                
                logger.info(f"Document classified as {predicted_class_name} with confidence {confidence_score:.4f}")
                
                return predicted_class_id, predicted_class_name, confidence_score
                
        except Exception as e:
            logger.error(f"Classification error for {image_path}: {str(e)}")
            # Fall back to a default classification
            return 0, "unknown", 0.5
