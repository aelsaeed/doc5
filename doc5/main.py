#!/usr/bin/env python3
"""
Document Processor Application Entry Point
"""
import os
import logging
import webbrowser
import sys
import traceback
from threading import Thread

from flask import Flask, jsonify, request, current_app, render_template
from document_processor.config import Config
from document_processor.db.database import init_db
from document_processor.api.routes import register_api_routes
from document_processor.web.views import register_web_routes
from document_processor.web.error_handlers import register_error_handlers
from document_processor.utils.gpu_utils import check_gpu_availability
from document_processor.utils.custom_exceptions import DocumentProcessorError
from document_processor.core.processing_modes import ProcessingMode

def configure_logging(app):
    """
    Configure application logging
    
    Args:
        app (Flask): Flask application
    """
    # Create logs directory if not exists
    log_dir = os.path.join(app.root_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up file handler with detailed formatting
    file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    ))
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []  # Remove any existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Logging configured")
    
    # Set library logging levels
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('org.apache.pdfbox').setLevel(logging.ERROR)  # Suppress PDFBox warnings
    
    return logger

def create_app(config_class=Config):
    """
    Factory function to create and configure the Flask application
    
    Args:
        config_class: Configuration class
        
    Returns:
        Flask: Configured Flask application
    """
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Calculate absolute paths for templates, uploads, and static files
    template_path = os.path.join(project_root, 'document_processor', 'web', 'templates')
    upload_path = os.path.join(project_root, 'uploads')
    static_path = os.path.join(project_root, 'static')
    
    # Log the paths to help with debugging
    print(f"Project root: {project_root}")
    print(f"Template path: {template_path}")
    print(f"Upload path: {upload_path}")
    print(f"Static path: {static_path}")
    
    # Initialize Flask app with absolute paths
    app = Flask(__name__, 
                template_folder=template_path,
                static_folder=static_path)
                
    # Update configuration with absolute paths
    app.config.from_object(config_class)
    app.config['UPLOAD_FOLDER'] = upload_path
    app.config['STATIC_FOLDER'] = static_path
    
    # Configure logging
    logger = configure_logging(app)
    logger.info(f"Template folder: {template_path}")
    logger.info(f"Upload folder: {upload_path}")
    logger.info(f"Static folder: {static_path}")
    
    try:
        # Check GPU availability
        device = check_gpu_availability()
        logger.info(f"Using device: {device}")
        
        # Ensure required directories exist
        for directory_key in ['UPLOAD_FOLDER', 'STATIC_FOLDER', 'MODELS_FOLDER', 'LOGS_FOLDER']:
            if directory_key in app.config:
                directory_path = app.config[directory_key]
                if not os.path.exists(directory_path):
                    logger.info(f"Creating directory: {directory_path}")
                    os.makedirs(directory_path, exist_ok=True)
            else:
                logger.warning(f"Configuration key {directory_key} not found in app config")
        
        # Initialize database
        init_db(app)
        
        # Register routes
        register_api_routes(app)
        register_web_routes(app)
        
        # Register error handlers
        register_error_handlers(app)
        
        # Add request logging middleware
        @app.before_request
        def log_request_info():
            logger.debug(f"Request: {request.method} {request.path}")
            if request.is_json:
                logger.debug(f"JSON data: {request.get_json()}")
        
        # Add global exception handler for unhandled exceptions
        @app.errorhandler(Exception)
        def handle_exception(e):
            logger.error(f"Unhandled exception: {str(e)}")
            logger.error(traceback.format_exc())
            
            if isinstance(e, DocumentProcessorError):
                # Let the specific error handler handle it
                raise e
                
            if request.path.startswith('/api/'):
                return jsonify({
                    'error': 'Internal Server Error',
                    'message': 'An unexpected error occurred'
                }), 500
            
            # For web routes, render the 500 error page
            return render_template('errors/500.html'), 500

        # Create a function to run tasks with application context
        def run_with_context(func, *args, **kwargs):
            try:
                # Check if we're already in an application context
                current_app._get_current_object()
            except RuntimeError:
                # If not, create a new one
                with app.app_context():
                    return func(*args, **kwargs)
            else:
                # If we are, just run the function
                return func(*args, **kwargs)

        # Store the context runner and app instance in config for use in other modules
        app.config['run_with_context'] = run_with_context
        app.config['_app_instance'] = app
        
        return app
        
    except Exception as e:
        logger.error(f"Error during app initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def run_app(host='0.0.0.0', port=5000, open_browser=True):
    """
    Run the application
    
    Args:
        host (str): Host to bind to
        port (int): Port to bind to
        open_browser (bool): Whether to open a browser window
    """
    try:
        app = create_app()
        
        # Open browser if requested
        if open_browser:
            url = f'http://localhost:{port}'
            webbrowser.open(url)
        
        # Run app
        app.run(host=host, port=port, debug=Config.DEBUG)
        
    except Exception as e:
        logging.error(f"Failed to start application: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    run_app()