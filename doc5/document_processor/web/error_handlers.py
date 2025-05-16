"""
Error handlers for the web application
"""
import logging
from flask import render_template, request, jsonify
from werkzeug.exceptions import HTTPException

from document_processor.utils.custom_exceptions import DocumentProcessorError

logger = logging.getLogger(__name__)

def register_error_handlers(app):
    """
    Register error handlers with the Flask application
    
    Args:
        app (Flask): Flask application
    """
    @app.errorhandler(404)
    def not_found_error(error):
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Not Found',
                'message': 'The requested resource was not found'
            }), 404
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {str(error)}")
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred'
            }), 500
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(DocumentProcessorError)
    def handle_document_processor_error(error):
        logger.error(f"Document processor error: {str(error)}")
        if request.path.startswith('/api/'):
            return jsonify({
                'error': error.__class__.__name__,
                'message': str(error)
            }), 400
        return render_template('errors/error.html', error=error), 400
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.error(f"Unhandled exception: {str(error)}")
        
        # Handle HTTP exceptions
        if isinstance(error, HTTPException):
            if request.path.startswith('/api/'):
                return jsonify({
                    'error': error.name,
                    'message': error.description
                }), error.code
            return render_template('errors/error.html', error=error), error.code
        
        # Handle other exceptions
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred'
            }), 500
        return render_template('errors/500.html'), 500