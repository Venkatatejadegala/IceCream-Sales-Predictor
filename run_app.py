"""
Script to run the Streamlit app and automatically open in Chrome browser.
"""
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def open_browser():
    """Open the Streamlit app in Chrome after a short delay."""
    time.sleep(2)  # Wait for Streamlit to start
    url = "http://localhost:8501"
    # Try to open in Chrome specifically
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Users\{}\AppData\Local\Google\Chrome\Application\chrome.exe".format(
            Path.home().name
        ),
    ]
    
    chrome_found = False
    for chrome_path in chrome_paths:
        if Path(chrome_path).exists():
            webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
            webbrowser.get('chrome').open(url)
            chrome_found = True
            print(f"Opening in Chrome: {url}")
            break
    
    if not chrome_found:
        # Fallback to default browser
        webbrowser.open(url)
        print(f"Opening in default browser: {url}")

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = Path(__file__).parent
    app_path = script_dir / "app.py"
    
    if not app_path.exists():
        print(f"Error: {app_path} not found!")
        sys.exit(1)
    
    print("Starting Ice Cream Sales Predictor...")
    print("The app will open automatically in your browser.")
    print("Press Ctrl+C to stop the server.")
    print("-" * 50)
    
    # Start browser opening in a separate thread
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

