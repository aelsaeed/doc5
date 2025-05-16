"""
Script to set up the document processor project environment
"""
import os
import sys
import shutil
import subprocess
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Project directories
PROJECT_DIRS = [
    "document_processor",
    "document_processor/api",
    "document_processor/core",
    "document_processor/core/classification",
    "document_processor/core/extraction",
    "document_processor/core/information",
    "document_processor/db",
    "document_processor/utils",
    "document_processor/web",
    "document_processor/web/templates",
    "document_processor/web/templates/errors",
    "document_processor/uploads",
    "document_processor/models",
    "document_processor/logs",
]

def ensure_directory_exists(directory_path):
    """Ensure a directory exists, create it if it doesn't"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Created directory: {directory_path}")
    else:
        logger.info(f"Directory already exists: {directory_path}")

def create_empty_init_file(dir_path):
    """Create an empty __init__.py file in the specified directory"""
    init_file = os.path.join(dir_path, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            pass
        logger.info(f"Created __init__.py file in {dir_path}")

def setup_project_structure():
    """Set up the project directory structure and __init__.py files"""
    logger.info("Setting up project directory structure...")
    
    # Create directories
    for dir_path in PROJECT_DIRS:
        ensure_directory_exists(dir_path)
    
    # Create __init__.py files in Python package directories
    for dir_path in PROJECT_DIRS:
        if not dir_path.endswith(("uploads", "models", "logs", "templates", "errors")):
            create_empty_init_file(dir_path)
    
    logger.info("Project directory structure set up successfully")

def install_requirements():
    """Install project requirements"""
    logger.info("Installing project requirements...")
    
    try:
        # Check if pip is available
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True)
        
        # Install requirements
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        logger.info("Project requirements installed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        logger.error("Please install the requirements manually: pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"Error during requirements installation: {e}")

def install_project():
    """Install the project in development mode"""
    logger.info("Installing project in development mode...")
    
    try:
        # Install in development mode
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], check=True)
        
        logger.info("Project installed successfully in development mode")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install project: {e}")
        logger.error("Please install the project manually: pip install -e .")
    except Exception as e:
        logger.error(f"Error during project installation: {e}")

def check_torch_installation():
    """Check if PyTorch is installed with GPU support"""
    logger.info("Checking PyTorch installation...")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU device count: {torch.cuda.device_count()}")
            logger.info(f"GPU device name: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA is not available. The application will run on CPU only.")
    except ImportError:
        logger.error("PyTorch is not installed. Please install it with: pip install torch")
    except Exception as e:
        logger.error(f"Error checking PyTorch installation: {e}")

def check_transformers_installation():
    """Check if Transformers is installed"""
    logger.info("Checking Transformers installation...")
    
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("Transformers is not installed. Please install it with: pip install transformers")
    except Exception as e:
        logger.error(f"Error checking Transformers installation: {e}")

def run_tests():
    """Run project tests"""
    logger.info("Running project tests...")
    
    try:
        # Run tests
        result = subprocess.run([
            sys.executable, "-m", "pytest", "-v"
        ], capture_output=True, text=True)
        
        logger.info("Test results:")
        logger.info(result.stdout)
        
        if result.returncode != 0:
            logger.warning("Some tests failed. Please check the test output.")
        else:
            logger.info("All tests passed!")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run tests: {e}")
    except Exception as e:
        logger.error(f"Error during test execution: {e}")

def main():
    """Main function"""
    logger.info("Starting project setup...")
    
    setup_project_structure()
    install_requirements()
    install_project()
    check_torch_installation()
    check_transformers_installation()
    
    # Uncomment to run tests
    # run_tests()
    
    logger.info("Project setup completed!")
    logger.info("You can now run the application with: python main.py")

if __name__ == "__main__":
    main()
