#!/usr/bin/env python3
"""
Test script to verify template locations
"""
import os
import sys

def check_template_paths():
    """Check if templates exist at various possible paths"""
    # Get current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Define possible template paths
    possible_paths = [
        os.path.join(cwd, 'document_processor', 'web', 'templates', 'index.html'),
        os.path.join(cwd, 'web', 'templates', 'index.html'),
        os.path.join(cwd, 'templates', 'index.html'),
        os.path.join(os.path.dirname(cwd), 'document_processor', 'web', 'templates', 'index.html')
    ]
    
    # Check each path
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found template at: {path}")
        else:
            print(f"❌ No template at: {path}")
    
    # List all files in the current directory (recursive, up to 2 levels)
    print("\nSearching for template files in current directory...")
    for root, dirs, files in os.walk(cwd):
        if root.count(os.sep) - cwd.count(os.sep) <= 2:  # Limit depth to 2 levels
            for file in files:
                if file.endswith('.html'):
                    print(f"Found HTML file: {os.path.join(root, file)}")

if __name__ == "__main__":
    check_template_paths()
