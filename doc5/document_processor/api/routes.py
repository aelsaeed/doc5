"""
API routes for the document processor application
"""
import os
import logging
from werkzeug.utils import secure_filename
from flask import Blueprint, jsonify, request, current_app

from document_processor.core.processor_service import ProcessorService
from document_processor.core.processing_modes import ProcessingMode
from document_processor.utils.file_utils import create_unique_filename, is_valid_document

logger = logging.getLogger(__name__)

# Create a blueprint for the API routes
api_bp = Blueprint('api', __name__, url_prefix='/api')

def register_api_routes(app):
    """
    Register API routes with the Flask application
    
    Args:
        app (Flask): Flask application
    """
    app.register_blueprint(api_bp)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    
    Returns:
        Response: JSON response
    """
    return jsonify({
        'status': 'ok',
        'message': 'Document processor API is running'
    })

@api_bp.route('/process', methods=['POST'])
def process_document():
    """
    Process a document via API
    
    Returns:
        Response: JSON response with processing results
    """
    try:
        # Check if a file was uploaded
        if 'document' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please provide a document file'
            }), 400
            
        file = request.files['document']
        
        # Check if the file has a filename
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a document file'
            }), 400
        
        # Get processing mode
        processing_mode = request.form.get('processing_mode', 'traditional')
        mode = ProcessingMode.LAYOUTLM if processing_mode == 'layoutlm' else ProcessingMode.TRADITIONAL
            
        # Save and process the file
        if file:
            filename = secure_filename(file.filename)
            
            # Check if the file type is supported
            if not is_valid_document(filename):
                return jsonify({
                    'error': 'Unsupported file type',
                    'message': 'Please upload a PDF, Word document, or image file'
                }), 400
            
            # Create a unique filename
            unique_filename = create_unique_filename(filename)
            upload_folder = current_app.config['UPLOAD_FOLDER']
            file_path = os.path.join(upload_folder, unique_filename)
            
            # Save the file
            file.save(file_path)
            logger.info(f"File uploaded: {file_path}")
            
            # Process the document
            processor = ProcessorService(current_app.config)
            result = processor.process_document(file_path, mode=mode)
            
            # Return processing results
            return jsonify({
                'status': 'success',
                'message': 'Document processed successfully',
                'result': result
            })
        
    except DocumentProcessorError as e:
        return jsonify({
            'error': 'Processing error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error processing document: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred while processing the document'
        }), 500

@api_bp.route('/modes', methods=['GET'])
def get_processing_modes():
    """
    Get available processing modes
    
    Returns:
        Response: JSON response with available modes
    """
    modes = [
        {
            'id': 'traditional',
            'name': 'Traditional Processing',
            'description': 'Uses conventional OCR and NLP techniques for document processing'
        },
        {
            'id': 'layoutlm',
            'name': 'LayoutLMv3 Processing',
            'description': 'Uses advanced AI model that understands both text and document layout'
        }
    ]
    
    return jsonify({
        'status': 'success',
        'modes': modes
    })

@api_bp.route('/documents', methods=['GET'])
def list_documents():
    """
    List processed documents
    
    Returns:
        Response: JSON response with document list
    """
    # This would typically query the database for processed documents
    # For now, just return an empty list
    return jsonify({
        'status': 'success',
        'documents': []
    })

@api_bp.route('/document/<document_id>', methods=['GET'])
def get_document(document_id):
    """
    Get a specific document by ID
    
    Args:
        document_id (str): Document ID
        
    Returns:
        Response: JSON response with document details
    """
    # This would typically query the database for a specific document
    # For now, just return a not found error
    return jsonify({
        'error': 'Not found',
        'message': f'Document with ID {document_id} not found'
    }), 404