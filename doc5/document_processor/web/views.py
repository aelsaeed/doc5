"""
Web views for the document processor application
"""
import os
import logging
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app, jsonify, session, send_from_directory

from document_processor.core.processor_service import ProcessorService
from document_processor.core.processing_modes import ProcessingMode
from document_processor.utils.custom_exceptions import DocumentProcessorError, ProcessingError
from document_processor.utils.file_utils import create_unique_filename, is_valid_document

logger = logging.getLogger(__name__)

# Create a blueprint for the web views
web_bp = Blueprint('web', __name__, template_folder='templates')

def register_web_routes(app):
    """
    Register web routes with the Flask application
    
    Args:
        app (Flask): Flask application
    """
    app.register_blueprint(web_bp)

@web_bp.route('/')
def index():
    """
    Render the homepage
    
    Returns:
        Response: Rendered template
    """
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index template: {str(e)}")
        return f"Error rendering template: {str(e)}", 500
    
@web_bp.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    Handle document upload
    
    Returns:
        Response: Rendered template or redirect
    """
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'document' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
            
        file = request.files['document']
        
        # Check if the file has a filename
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        # Get selected processing mode
        processing_mode = request.form.get('processing_mode', 'bulk')
        mode = ProcessingMode.TARGETED if processing_mode == 'targeted' else ProcessingMode.BULK
        
        # Store mode in session for future uploads
        session['processing_mode'] = processing_mode
            
        # Save and process the file
        if file:
            try:
                filename = secure_filename(file.filename)
                
                # Check if the file type is supported
                if not is_valid_document(filename):
                    flash('File type not supported. Please upload a PDF, Word document, or image file.', 'error')
                    return redirect(request.url)
                
                # Create a unique filename
                unique_filename = create_unique_filename(filename)
                upload_folder = current_app.config['UPLOAD_FOLDER']
                file_path = os.path.join(upload_folder, unique_filename)
                
                # Ensure upload folder exists
                os.makedirs(upload_folder, exist_ok=True)
                
                # Save the file
                file.save(file_path)
                logger.info(f"File uploaded: {file_path}")
                
                # Redirect to the processing page
                return redirect(url_for('web.process_document', 
                                       file_path=unique_filename, 
                                       mode=mode.value))
                                       
            except Exception as e:
                logger.error(f"File upload error: {str(e)}")
                flash(f"Error uploading file: {str(e)}", 'error')
                return redirect(request.url)
    
    # GET request - show upload form
    # Get previously selected mode from session or default to bulk
    selected_mode = session.get('processing_mode', 'bulk')
    
    return render_template('upload.html', selected_mode=selected_mode)

@web_bp.route('/process/<file_path>')
def process_document(file_path):
    """
    Process a document and display the results
    
    Args:
        file_path (str): Path to the document (filename only)
        
    Returns:
        Response: Rendered template
    """
    try:
        # Get processing mode
        mode_str = request.args.get('mode', 'bulk')
        mode = ProcessingMode.TARGETED if mode_str == 'targeted' else ProcessingMode.BULK
        
        # Get full file path
        upload_folder = current_app.config['UPLOAD_FOLDER']
        full_file_path = os.path.join(upload_folder, file_path)
        
        # Check if file exists
        if not os.path.exists(full_file_path):
            flash('File not found', 'error')
            return redirect(url_for('web.upload'))
        
        # Create processor service
        processor = ProcessorService(current_app.config)
        
        # Process the document
        result = processor.process_document(full_file_path, mode=mode)
        
        # Render results
        return render_template('results.html', result=result, mode=mode.value)
        
    except ProcessingError as e:
        flash(str(e), 'error')
        return redirect(url_for('web.upload'))
    except DocumentProcessorError as e:
        flash(str(e), 'error')
        return redirect(url_for('web.upload'))
    except Exception as e:
        logger.error(f"Unexpected error processing document: {str(e)}")
        flash('An unexpected error occurred while processing the document', 'error')
        return redirect(url_for('web.upload'))

@web_bp.route('/documents')
def list_documents():
    """
    List processed documents
    
    Returns:
        Response: Rendered template
    """
    # This would typically query the database for processed documents
    # For now, just return a simple template
    return render_template('documents.html', documents=[])

@web_bp.route('/settings')
def settings():
    """
    Application settings page
    
    Returns:
        Response: Rendered template
    """
    return render_template('settings.html')

@web_bp.route('/visualization/<path:visualization_path>')
def view_visualization(visualization_path):
    """
    Display bounding box visualization
    
    Args:
        visualization_path (str): Path to visualization file
        
    Returns:
        Response: Rendered template with visualization
    """
    try:
        # Get static folder
        static_folder = current_app.config['STATIC_FOLDER']
        
        # Check if the file exists
        full_path = os.path.join(static_folder, visualization_path)
        if not os.path.exists(full_path):
            logger.error(f"Visualization file not found: {full_path}")
            flash('Visualization file not found', 'error')
            return redirect(url_for('web.index'))
        
        # If it's a direct image request
        if request.headers.get('Accept', '').startswith('image/'):
            return send_from_directory(static_folder, visualization_path)
        
        # Otherwise render the visualization template
        return render_template('visualization.html', image_path=visualization_path)
    except Exception as e:
        logger.error(f"Error displaying visualization: {str(e)}")
        flash(f"Error displaying visualization: {str(e)}", 'error')
        return redirect(url_for('web.index'))

@web_bp.route('/static/<path:filename>')
def serve_static(filename):
    """
    Serve static files
    
    Args:
        filename (str): Path to static file
        
    Returns:
        Response: Static file
    """
    static_folder = current_app.config['STATIC_FOLDER']
    logger.debug(f"Serving static file: {filename} from folder: {static_folder}")
    return send_from_directory(static_folder, filename)