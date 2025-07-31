#!/bin/bash

# Production startup script for Ezra backend

echo "Starting Ezra Backend in Production Mode..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Install dependencies
pip install -r requirements.txt

# Start with gunicorn for production
gunicorn --bind 0.0.0.0:$PORT --workers 4 --timeout 120 server:app 