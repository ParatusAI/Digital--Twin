#\!/bin/bash
# Set up Python environment for CsPbBr3 Digital Twin

# Add local packages to Python path
export PYTHONPATH="/home/aroyston/.local/lib/python3.8/site-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"

# Run the command with correct path
exec "$@"
