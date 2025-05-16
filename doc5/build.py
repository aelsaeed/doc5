import os
import sys

# Define the project structure
project_structure = {
    "document_processor": {
        "__init__.py": "",
        "config.py": "config-py",
        "main.py": "main-py-update",
        "db": {
            "__init__.py": "",
            "database.py": "database-py",
            "models.py": "models-py"
        },
        "api": {
            "__init__.py": "",
            "routes.py": "api-routes-update"
        },
        "core": {
            "__init__.py": "",
            "processor_service.py": "processor-service-update",
            "classification": {
                "__init__.py": "",
                "classifier.py": "classifier-py"
            },
            "extraction": {
                "__init__.py": "",
                "text_extractor.py": "text-extractor-update"
            },
            "information": {
                "__init__.py": "",
                "financial_extractor.py": "financial-extractor-py",
                "date_extractor.py": "date-extractor-py"
            }
        },
        "utils": {
            "__init__.py": "",
            "custom_exceptions.py": "custom-exceptions-py",
            "file_utils.py": "file-utils-py",
            "gpu_utils.py": "gpu-utils-py",
            "validation.py": "validation-py"
        },
        "web": {
            "__init__.py": "",
            "error_handlers.py": "error-handlers-py",
            "views.py": "web-views-update"
        },
        "tests": {
            "__init__.py": "",
            "test_base.py": "test-base-py",
            "test_classifier.py": "test-classifier-py",
            "test_extractor.py": "test-extractor-py",
            "test_financial_extractor.py": "test-financial-extractor-py",
            "test_processor.py": "test-processor-py",
            "test_database.py": "test-database-py",
            "test_api_integration.py": "test-api-integration"
        }
    }
}

# Create directories
def create_directories(structure, base_path=""):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_directories(content, path)
        else:
            # Skip files here, we'll create them in the next function
            pass

# Create files
def create_files(structure, base_path=""):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            create_files(content, path)
        else:
            # Create file with content
            with open(path, "w") as f:
                f.write(content)

# Main function to set up the project
def setup_project():
    print("Creating project structure...")
    create_directories(project_structure)
    create_files(project_structure)
    
    # Create additional directories
    os.makedirs("document_processor/uploads", exist_ok=True)
    os.makedirs("document_processor/logs", exist_ok=True)
    
    # Create requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("""flask==3.0.3
pymupdf==1.24.10
tabula-py==2.9.3
spacy==3.8.0
python-dateutil==2.9.0
sentence-transformers==3.2.0
pillow==10.4.0
transformers==4.44.2
docling
pytest
pytest-cov
pytest-mock
""")
    
    # Create setup.py
    with open("setup.py", "w") as f:
        f.write("""from setuptools import setup, find_packages

setup(
    name="document_processor",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask",
        "pymupdf",
        "tabula-py",
        "spacy",
        "python-dateutil",
        "sentence-transformers",
        "pillow",
        "transformers",
        "docling",
    ],
)
""")
    
    print("Project structure created successfully!")
    print("Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Install the project in development mode: pip install -e .")
    print("3. Download spaCy model: python -m spacy download en_core_web_sm")
    print("4. Run the application: python -m document_processor.main")

if __name__ == "__main__":
    setup_project()