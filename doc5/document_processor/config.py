"""
Configuration settings for the document processor application
"""
import os
from pathlib import Path

class Config:
    """Base configuration class"""
    # Application settings
    DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    # File paths
    BASE_DIR = Path(__file__).resolve().parent
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', str(BASE_DIR / 'uploads'))
    MODELS_FOLDER = os.environ.get('MODELS_FOLDER', str(BASE_DIR / 'models'))
    LOGS_FOLDER = os.environ.get('LOGS_FOLDER', str(BASE_DIR / 'logs'))
    
    # Database settings
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///' + str(BASE_DIR / 'document_processor.db'))
    
    # Processing settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'doc', 'docx', 'txt'}
    
    # Model settings
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    # In production, these should be set from environment variables
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DATABASE_URL = os.environ.get('DATABASE_URL')

# Set configuration based on environment
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])