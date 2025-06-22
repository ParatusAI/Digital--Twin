#!/bin/bash
export PATH="/home/aroyston/.local/bin:$PATH"
export PYTHONPATH="/usr/lib/python3/dist-packages:/home/aroyston/.local/lib/python3.8/site-packages:$PYTHONPATH"
streamlit run streamlit_app.py --server.port 8503