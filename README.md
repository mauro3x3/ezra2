# Ezra Backend API

AI-powered ancestry analysis backend for the Ezra mobile app.

## Features

- **Image Analysis**: Analyzes photos to determine ancestry breakdown
- **Health Insights**: Generates health-related insights based on ancestry
- **AI Chat**: Provides detailed health consultations via AI

## API Endpoints

### POST `/analyze`
Analyzes a photo and returns ancestry breakdown.

**Request:**
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "breakdown": [
    {"country": "Turkey", "percentage": 35.0},
    {"country": "Lebanon", "percentage": 25.0},
    {"country": "Iran", "percentage": 20.0},
    {"country": "Iraq", "percentage": 12.0},
    {"country": "Israel", "percentage": 8.0}
  ],
  "health_insights": "Health insights based on ancestry..."
}
```

### POST `/chat`
Chat with AI doctor about health traits.

**Request:**
```json
{
  "trait": "Lactose Intolerance",
  "userMessage": "What does this mean for my diet?"
}
```

**Response:**
```json
{
  "response": "AI doctor's response..."
}
```

### GET `/health`
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-31T12:00:00",
  "version": "1.0.0"
}
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `FLASK_ENV`: Set to "production" for production
- `PORT`: Server port (default: 5001)

## Deployment

### Docker
```bash
docker build -t ezra-backend .
docker run -p 5001:5001 ezra-backend
```

### Railway
1. Connect GitHub repository
2. Add environment variables
3. Deploy automatically

### Local Development
```bash
cd backend
source venv/bin/activate
python server.py
```

## Technologies

- **Python 3.11**
- **Flask** - Web framework
- **OpenCV** - Image processing
- **OpenAI GPT-4** - AI analysis
- **DeepFace** - Face detection 