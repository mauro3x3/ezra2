#!/bin/bash

echo "Setting up Ezra Backend with ChatGPT Integration..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "⚠️  WARNING: OPENAI_API_KEY environment variable is not set."
    echo "To enable ChatGPT health insights, set your OpenAI API key:"
    echo ""
    echo "export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "You can get an API key from: https://platform.openai.com/api-keys"
    echo ""
    echo "The app will still work without it, but health insights will be unavailable."
    echo ""
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the backend server:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the server: python server.py"
echo ""
echo "The server will be available at: http://127.0.0.1:5000" 