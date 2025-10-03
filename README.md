# Script Connector Bot

A smart tool that analyzes video scripts to identify missing connectors and suggests contextual improvements.

## Features

- **Smart Script Analysis**: Detects intro, sections, payoff, and existing connectors
- **AI-Powered Suggestions**: Generates contextual connector suggestions using OpenAI
- **Visual Placement**: Shows exactly where to place connectors in the script
- **PDF Support**: Upload and analyze PDF scripts
- **Intro Integration**: Uses custom intro to generate story-based connectors

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open http://localhost:5001

## Deployment on Vercel

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Set environment variables in Vercel dashboard:
   - `OPENAI_API_KEY` (optional, for AI-powered suggestions)

4. Deploy!

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for AI-powered connector suggestions (optional)

## Usage

1. Upload a PDF script or paste text
2. Optionally add your script's intro for better suggestions
3. Click "Analyze Script"
4. Review suggestions and click to see placement in the script

**Note:** OpenAI API key should be set as an environment variable in Vercel for AI-powered suggestions.

## API Endpoints

- `POST /analyze` - Analyze a script
- `POST /upload_pdf` - Upload and extract text from PDF
- `POST /suggest_connector` - Generate connector suggestions
- `POST /segment_text` - AI-powered text segmentation
