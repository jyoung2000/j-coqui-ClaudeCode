#!/usr/bin/env python3
"""
Startup script for Coqui TTS Web Server
Handles environment setup and numba cache issues
"""

import os
import sys
import tempfile

# Set environment variables to fix numba caching issues
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_JIT'] = '0'

# Create cache directory if it doesn't exist
cache_dir = os.environ.get('NUMBA_CACHE_DIR', '/tmp/numba_cache')
os.makedirs(cache_dir, exist_ok=True)

# Set permissions for cache directory
try:
    os.chmod(cache_dir, 0o755)
except:
    pass

# Import and run the main application
if __name__ == '__main__':
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Import the main app
    from app import app, config, logger
    
    logger.info("Starting Enhanced Coqui TTS Web Server with fixed environment...")
    logger.info(f"NUMBA_CACHE_DIR: {os.environ.get('NUMBA_CACHE_DIR')}")
    logger.info(f"Server starting on port {config.PORT}")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=config.PORT,
        debug=False,
        threaded=True
    )