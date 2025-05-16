# Use your existing image as the base
FROM ahmedelsaeed833/dhap-document-processor:latest

# Set working directory
WORKDIR /app

# Copy your application files to make sure everything is in the right place
COPY main.py /app/
COPY document_processor/ /app/document_processor/
COPY static/ /app/static/
COPY requirements.txt /app/

# Install ALL dependencies from requirements.txt to ensure nothing is missing
RUN pip install --no-cache-dir -r requirements.txt \
    # Install the specific packages you mentioned
    && pip install --no-cache-dir python_doctr==0.11.0 sentence_transformers==3.2.0 \
    # Make sure SQLAlchemy is explicitly installed
    && pip install --no-cache-dir SQLAlchemy==2.0.40 \
    # Clean up all caches to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip /root/.cache/torch /root/.cache/huggingface

# Create necessary directories in case they don't exist
RUN mkdir -p /app/logs /app/uploads /app/static

# Make sure port 5000 is properly exposed
EXPOSE 5000

# Run the application directly
CMD ["python", "/app/main.py"]