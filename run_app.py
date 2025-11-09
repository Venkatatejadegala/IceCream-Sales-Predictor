# run_app.py
# Auto-launcher for Streamlit application

import os
import webbrowser
import subprocess
import time

# Streamlit app file name
APP_FILE = "app.py"

def run_app():
    """Runs Streamlit app and opens it in Chrome automatically."""
    print("🚀 Starting Streamlit app...")

    # Launch Streamlit app in background
    subprocess.Popen(["streamlit", "run", APP_FILE])

    # Wait a moment for server to start
    time.sleep(3)

    # Open in browser (Chrome preferred)
    url = "http://localhost:8501"
    try:
        webbrowser.get("chrome").open(url)
    except:
        # fallback if Chrome is not available
        webbrowser.open(url)
    print("✅ Application started successfully at:", url)

if __name__ == "__main__":
    run_app()
