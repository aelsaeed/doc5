"""
Database models for the document processor application
"""
import datetime
import json
import uuid
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB

from document_processor.core.processing_modes import ProcessingMode

Base = declarative_base()

class Document(Base):
    """Document model representing a processed document"""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    
    # Document classification
    doc_type = Column(String(100), nullable=True)
    classification_confidence = Column(Float, nullable=True)
    
    # Processing details
    processing_mode = Column(Enum(ProcessingMode), nullable=False, default=ProcessingMode.BULK)
    processing_time = Column(Float, nullable=True)  # In seconds
    
    # Content
    text_content = Column(Text, nullable=True)
    
    # Metadata
    doc_metadata = Column(JSONB, nullable=True)
    
    # Status
    processing_status = Column(String(50), default='pending')
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    entities = relationship("Entity", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', type='{self.doc_type}')>"
    
    def to_dict(self):
        """Convert document to dictionary"""
        return {
            'id': self.id,
            'filename': self.filename,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'doc_type': self.doc_type,
            'classification_confidence': self.classification_confidence,
            'processing_mode': self.processing_mode.value,
            'processing_time': self.processing_time,
            'processing_status': self.processing_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'entities': [entity.to_dict() for entity in self.entities] if self.entities else []
        }

class Entity(Base):
    """Entity model representing an extracted entity from a document"""
    __tablename__ = 'entities'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    
    # Entity details
    entity_type = Column(String(100), nullable=False)
    text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    field = Column(String(100), nullable=True)  # Field name for targeted extraction
    
    # Position in document (for visual entities)
    page_num = Column(Integer, nullable=True)
    start_idx = Column(Integer, nullable=True)
    end_idx = Column(Integer, nullable=True)
    
    # For LayoutLM entities
    bounding_box = Column(JSONB, nullable=True)  # JSON with coordinates: [x0, y0, x1, y1]
    
    # Relationships
    document = relationship("Document", back_populates="entities")
    
    def __repr__(self):
        return f"<Entity(id={self.id}, type='{self.entity_type}', text='{self.text[:20]}...')>"
    
    def to_dict(self):
        """Convert entity to dictionary"""
        return {
            'id': self.id,
            'entity_type': self.entity_type,
            'text': self.text,
            'field': self.field,
            'confidence': self.confidence,
            'page_num': self.page_num,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'bounding_box': self.bounding_box
        }

# Utility functions
def generate_session_id():
    """Generate a unique session ID for document processing"""
    return str(uuid.uuid4())

class PartnershipDetails:
    """Data class for partnership details"""
    def __init__(self, name=None, address=None, ein=None, recipient_name=None, recipient_address=None):
        self.name = name
        self.address = address
        self.ein = ein
        self.recipient_name = recipient_name
        self.recipient_address = recipient_address

class FinancialInformation:
    """Data class for financial information"""
    def __init__(self, net_income=None, portfolio_income=None):
        self.net_income = net_income
        self.portfolio_income = portfolio_income

class ImportantDate:
    """Data class for important dates"""
    def __init__(self, event=None, date=None):
        self.event = event
        self.date = date

class Investment:
    """Data class for investment information"""
    def __init__(self, id=None, name=None, type=None):
        self.id = id
        self.name = name
        self.type = type