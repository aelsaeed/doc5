"""
Custom exceptions for the document processor application
"""

class DocumentProcessorError(Exception):
    """Base exception for all document processor errors"""
    pass

class DocumentNotSupportedError(DocumentProcessorError):
    """Exception raised when document format is not supported"""
    pass

class ProcessingError(DocumentProcessorError):
    """Exception raised when document processing fails"""
    pass

class ClassificationError(DocumentProcessorError):
    """Exception raised when document classification fails"""
    pass

class TextExtractionError(DocumentProcessorError):
    """Exception raised when text extraction fails"""
    pass

class ModelLoadError(DocumentProcessorError):
    """Exception raised when model loading fails"""
    pass

class DatabaseError(DocumentProcessorError):
    """Exception raised when database operations fail"""
    pass

class ValidationError(DocumentProcessorError):
    """Exception raised when validation fails"""
    pass

class ConfigurationError(DocumentProcessorError):
    """Exception raised when configuration is invalid"""
    pass

class AuthenticationError(DocumentProcessorError):
    """Exception raised when authentication fails"""
    pass

class AuthorizationError(DocumentProcessorError):
    """Exception raised when authorization fails"""
    pass

class ResourceNotFoundError(DocumentProcessorError):
    """Exception raised when a resource is not found"""
    pass

class DuplicateResourceError(DocumentProcessorError):
    """Exception raised when a duplicate resource is detected"""
    pass

class APIError(DocumentProcessorError):
    """Exception raised when an API error occurs"""
    pass

class ThirdPartyServiceError(DocumentProcessorError):
    """Exception raised when a third-party service error occurs"""
    pass

class FileTypeError(DocumentProcessorError):
    """Exception raised when a third-party service error occurs"""
    pass

class FileReadError(DocumentProcessorError):
    """Exception raised when a third-party service error occurs"""
    pass

class EmptyTextError(DocumentProcessorError):
    """Exception raised when a third-party service error occurs"""
    pass

class FileSizeError(DocumentProcessorError):
    """Exception raised when a third-party service error occurs"""
    pass

class ExtractionError(DocumentProcessorError):
    """Exception raised when a third-party service error occurs"""
    pass