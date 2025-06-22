#!/usr/bin/env python3
"""
Heroku-compatible Web Interface for CsPbBr₃ Digital Twin
"""

import os
from web_interface import app, initialize_system

if __name__ == "__main__":
    # Initialize the system
    initialize_system()
    
    # Get port from environment (Heroku requirement)
    port = int(os.environ.get("PORT", 5000))
    
    # Run with production settings
    app.run(host="0.0.0.0", port=port, debug=False)
