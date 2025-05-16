"""
Text extraction module for various document types
"""
import os
import logging
import traceback
from typing import Dict, List, Tuple, Union, Any, Optional
import fitz  # PyMuPDF
import tabula
from PIL import Image

from document_processor.utils.custom_exceptions import (
    FileTypeError, FileReadError, TextExtractionError, EmptyTextError
)
from document_processor.utils.validation import validate_file
from document_processor.utils.file_utils import get_file_extension
logger = logging.getLogger(__name__)

class BaseExtractor:
    """Base class for document extractors"""
    
    def __init__(self):
        """Initialize text extractor"""
        logger.info("Text extractor initialized")
    
    def extract_text(self, file_path):
        """
        Extract text from document
        
        Args:
            file_path (str): Path to document
            
        Returns:
            str: Extracted text
        """
        # Get file extension
        ext = get_file_extension(file_path).lower()
        
        # Extract text based on file type
        if ext == 'pdf':
            return self._extract_from_pdf(file_path)
        elif ext in ['jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp']:
            return self._extract_from_image(file_path)
        elif ext in ['txt', 'text']:
            return self._extract_from_txt(file_path)
        elif ext in ['docx', 'doc']:
            return self._extract_from_word(file_path)
        else:
            logger.warning(f"Unsupported file format for text extraction: {ext}")
            return ""
    
    def _extract_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            text = ""
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""
    
    def _extract_from_txt(self, file_path):
        """Extract text from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error extracting text from text file {file_path}: {str(e)}")
            return ""
    
    def _extract_from_image(self, file_path):
        """Extract text from image file"""
        # This would use OCR in a real implementation
        logger.warning("Image OCR not implemented in this version")
        return f"[Text extracted from image: {os.path.basename(file_path)}]"
    
    def _extract_from_word(self, file_path):
        """Extract text from Word document"""
        # This would use a library like python-docx in a real implementation
        logger.warning("Word document extraction not implemented in this version")
        return f"[Text extracted from Word document: {os.path.basename(file_path)}]"

    def _get_file_extension(self, file_path):
        """Get file extension"""
        return os.path.splitext(file_path)[1].lower()

class PdfExtractor(BaseExtractor):
    """Extractor for PDF documents"""
    
    def __init__(self):
        """Initialize the PDF extractor"""
        super().__init__()
        self.supported_extensions = ['.pdf']
        self.min_text_length = 100  # Minimum characters to consider extraction successful
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and tables from a PDF document
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted text and tables
            
        Raises:
            FileTypeError: When file is not a PDF
            FileReadError: When file cannot be read
            TextExtractionError: When text extraction fails
            EmptyTextError: When extracted text is empty or minimal
        """
        try:
            # Validate file
            result, error = validate_file(file_path, self.supported_extensions)
            if not result:
                raise error
            
            # Extract text using PyMuPDF
            text = self._extract_text(file_path)
            
            # Check if we got meaningful text
            if len(text.strip()) < self.min_text_length:
                logger.warning(f"Minimal text extracted from {file_path}, likely a scanned PDF")
                try:
                    # Try Docling as fallback
                    logger.info("Trying Docling for PDF extraction due to minimal text")
                    result = self._extract_with_docling(file_path)
                    
                    # Check if Docling extraction was successful
                    if len(result["text"].strip()) < self.min_text_length:
                        logger.warning("Docling also extracted minimal text")
                        # Return original text if Docling didn't improve results
                        if len(result["text"].strip()) < len(text.strip()):
                            result["text"] = text
                    
                    return result
                except ImportError:
                    logger.warning("Docling not installed. Continuing with minimal text.")
                except Exception as docling_error:
                    logger.warning(f"Docling extraction failed: {str(docling_error)}")
                    # Proceed with minimal text, but log a warning
            
            # Extract tables using tabula
            tables = self._extract_tables(file_path)
            
            return {
                "text": text,
                "tables": tables
            }
        except FileTypeError as e:
            # Re-raise file type errors
            raise
        except FileReadError as e:
            # Re-raise file read errors
            raise
        except Exception as e:
            logger.error(f"Error extracting from PDF {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            
            try:
                # Fallback to Docling if available
                logger.info("Trying Docling as fallback for PDF extraction after error")
                return self._extract_with_docling(file_path)
            except ImportError:
                logger.error("Docling not installed. Install with `pip install docling`.")
                raise TextExtractionError(file_path, f"Primary extraction failed: {str(e)}, and Docling is not installed")
            except Exception as docling_error:
                logger.error(f"Docling extraction failed: {str(docling_error)}")
                logger.error(traceback.format_exc())
                raise TextExtractionError(file_path, f"Both extraction methods failed. Original error: {str(e)}, Fallback error: {str(docling_error)}", True)
    
    def _extract_text(self, file_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text
            
        Raises:
            TextExtractionError: When text extraction fails
        """
        try:
            text = ""
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PyMuPDF text extraction failed: {str(e)}")
            raise TextExtractionError(file_path, f"PyMuPDF error: {str(e)}")
    
    def _extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF using tabula
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            List[Dict[str, Any]]: List of tables as dictionaries
        """
        try:
            df_list = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
            tables = [df.to_dict(orient='records') for df in df_list if not df.empty]
            return tables
        except Exception as e:
            logger.warning(f"Table extraction failed: {str(e)}")
            # Don't raise exception for table extraction failures
            # Just return empty list and continue
            return []
    
    def _extract_with_docling(self, file_path: str) -> Dict[str, Any]:
        """
        Extract using Docling as a fallback
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted content
            
        Raises:
            ImportError: When Docling is not installed
            TextExtractionError: When Docling extraction fails
        """
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(file_path)
            text = result.document.export_to_markdown()
            tables = []
            
            if hasattr(result.document, 'tables') and result.document.tables:
                # Handle tables without relying on to_dict()
                for table in result.document.tables:
                    table_dict = {}
                    
                    # Try to extract header and rows data if available
                    if hasattr(table, 'header') and table.header:
                        table_dict['header'] = [str(h) for h in table.header]
                    
                    if hasattr(table, 'rows') and table.rows:
                        table_dict['rows'] = [[str(cell) for cell in row] for row in table.rows]
                    else:
                        # Alternative approach if rows aren't directly accessible
                        # Create a simple representation of the table
                        table_dict['data'] = str(table)
                    
                    tables.append(table_dict)
                
            return {
                "text": text,
                "tables": tables
            }
        except ImportError:
            logger.error("Docling not installed. Install with `pip install docling`.")
            raise
        except Exception as e:
            logger.error(f"Docling extraction failed: {str(e)}")
            raise TextExtractionError(file_path, f"Docling error: {str(e)}")

class ImageExtractor(BaseExtractor):
    """Extractor for image documents using Docling"""
    
    def __init__(self):
        """Initialize the image extractor"""
        super().__init__()
        self.supported_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from an image using Docling
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted text and tables
            
        Raises:
            FileTypeError: When file is not a supported image format
            FileReadError: When file cannot be read
            TextExtractionError: When text extraction fails
            EmptyTextError: When extracted text is empty or minimal
        """
        try:
            # Validate file
            result, error = validate_file(file_path, self.supported_extensions)
            if not result:
                raise error
            
            # Try Docling for image extraction
            try:
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                result = converter.convert(file_path)
                text = result.document.export_to_markdown()
                tables = []
                
                if hasattr(result.document, 'tables') and result.document.tables:
                    for table in result.document.tables:
                        table_dict = {}
                        if hasattr(table, 'header') and table.header:
                            table_dict['header'] = [str(h) for h in table.header]
                        if hasattr(table, 'rows') and table.rows:
                            table_dict['rows'] = [[str(cell) for cell in row] for row in table.rows]
                        else:
                            table_dict['data'] = str(table)
                        tables.append(table_dict)
                
                # Check if extraction was successful
                if not text.strip():
                    logger.warning(f"No text extracted by Docling for {file_path}")
                    return {"text": "", "tables": []}
                
                return {
                    "text": text,
                    "tables": tables
                }
            except ImportError:
                logger.warning("Docling not installed. Falling back to basic image extraction.")
                return {"text": "", "tables": []}
            except Exception as e:
                logger.warning(f"Docling failed for image extraction: {str(e)}. Falling back to basic extraction.")
                return {"text": "", "tables": []}
        except FileTypeError as e:
            raise
        except FileReadError as e:
            raise
        except Exception as e:
            logger.error(f"Image extraction failed for {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise TextExtractionError(file_path, f"Image extraction failed: {str(e)}")

class DocumentExtractorFactory:
    """Factory for creating document extractors based on file type"""
    
    @staticmethod
    def get_extractor(file_path: str) -> BaseExtractor:
        """
        Get the appropriate extractor for a document
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            BaseExtractor: An extractor instance for the document type
            
        Raises:
            FileTypeError: When file type is not supported
        """
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension == '.pdf':
            return PdfExtractor()
        elif extension in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
            return ImageExtractor()
        else:
            supported_types = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif']
            raise FileTypeError(file_path, supported_types)

def extract_document_content(file_path: str) -> Dict[str, Any]:
    """
    Extract content from a document file
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        Dict[str, Any]: Dictionary containing extracted content (text, tables)
        
    Raises:
        FileTypeError: When file type is not supported
        FileReadError: When file cannot be read
        TextExtractionError: When text extraction fails
        EmptyTextError: When extracted text is empty or minimal
    """
    try:
        # Validate file exists
        if not os.path.isfile(file_path):
            raise FileReadError(file_path, "File does not exist")
        
        # Get appropriate extractor
        extractor = DocumentExtractorFactory.get_extractor(file_path)
        
        # Extract content
        result = extractor.extract(file_path)
        
        # Check for empty content
        if not result.get("text", "").strip():
            raise EmptyTextError(file_path)
        
        return result
    except (FileTypeError, FileReadError, TextExtractionError, EmptyTextError):
        # Re-raise known exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in extract_document_content: {str(e)}")
        logger.error(traceback.format_exc())
        raise TextExtractionError(file_path, f"Unexpected error: {str(e)}")