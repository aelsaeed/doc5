"""
Database manager for the document processor application
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Create declarative base here
Base = declarative_base()

logger = logging.getLogger(__name__)

# Global session factory
db_session = None
engine = None

def init_db(app):
    """
    Initialize the database connection
    
    Args:
        app (Flask): Flask application
    """
    global db_session, engine
    
    try:
        # Create engine and session factory
        database_url = app.config.get('DATABASE_URL', 'sqlite:///document_processor.db')
        engine = create_engine(database_url)
        db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
        
        # Add query property to models
        Base.query = db_session.query_property()
        
        # Import models here to avoid circular imports
        from document_processor.db.models import Document, Entity
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)
        
        # Register session close on app shutdown
        @app.teardown_appcontext
        def shutdown_session(exception=None):
            if db_session:
                db_session.remove()
        
        logger.info(f"Database initialized with {database_url}")
        
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

def save_document_result(processing_result):
    """
    Save document processing results to database
    
    Args:
        processing_result (dict): Results from document processing
        
    Returns:
        Document: Saved document model
    """
    try:
        # Import here to avoid circular imports
        from document_processor.db.models import Document, Entity
        
        # Create document record
        document = Document(
            filename=processing_result.get('file_name'),
            file_path=processing_result.get('file_path'),
            file_type=processing_result.get('file_path', '').split('.')[-1],
            file_size=0,  # This would be calculated
            doc_type=processing_result.get('doc_type'),
            classification_confidence=processing_result.get('classification_confidence'),
            processing_mode=processing_result.get('processing_mode', 'traditional'),
            text_content=processing_result.get('text'),
            processing_status='completed'
        )
        
        # Add entities
        for entity_data in processing_result.get('entities', []):
            entity = Entity(
                entity_type=entity_data.get('type'),
                text=entity_data.get('text'),
                confidence=entity_data.get('confidence'),
                start_idx=entity_data.get('start_idx'),
                end_idx=entity_data.get('end_idx'),
                bounding_box=entity_data.get('bounding_box')
            )
            document.entities.append(entity)
        
        # Save to database
        db_session.add(document)
        db_session.commit()
        
        logger.info(f"Saved document {document.id} to database")
        return document
        
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error saving document to database: {str(e)}")
        raise

def get_document_by_id(document_id):
    """
    Get a document by ID
    
    Args:
        document_id (int): Document ID
        
    Returns:
        Document: Document model
    """
    from document_processor.db.models import Document
    return Document.query.get(document_id)

def get_documents(limit=50, offset=0, doc_type=None, processing_mode=None):
    """
    Get list of documents with optional filtering
    
    Args:
        limit (int): Maximum number of documents to return
        offset (int): Offset for pagination
        doc_type (str, optional): Filter by document type
        processing_mode (str, optional): Filter by processing mode
        
    Returns:
        list: List of documents
    """
    from document_processor.db.models import Document
    
    query = Document.query
    
    if doc_type:
        query = query.filter(Document.doc_type == doc_type)
    
    if processing_mode:
        query = query.filter(Document.processing_mode == processing_mode)
    
    query = query.order_by(Document.created_at.desc())
    return query.limit(limit).offset(offset).all()

def delete_document(document_id):
    """
    Delete a document by ID
    
    Args:
        document_id (int): Document ID
        
    Returns:
        bool: True if deleted, False otherwise
    """
    from document_processor.db.models import Document
    
    document = Document.query.get(document_id)
    
    if document:
        db_session.delete(document)
        db_session.commit()
        return True
    
    return False

def get_entity_by_id(entity_id):
    """
    Get an entity by ID
    
    Args:
        entity_id (int): Entity ID
        
    Returns:
        Entity: Entity model
    """
    from document_processor.db.models import Entity
    return Entity.query.get(entity_id)
