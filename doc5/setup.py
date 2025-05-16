from setuptools import setup, find_packages

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
        "doctr-io",
        "pdf2image",
        "torch",
        "torchvision",
    ],
)
