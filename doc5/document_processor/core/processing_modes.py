"""
Defines processing modes for the document processor
"""
from enum import Enum

class ProcessingMode(Enum):
    """Enum for document processing modes"""
    BULK = "bulk"      # Uses docling and PyMuPDF for bulk extraction
    TARGETED = "targeted"  # Uses doctr + LayoutLMv3 for targeted extraction