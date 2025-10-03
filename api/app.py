#!/usr/bin/env python3
"""
Web interface for Script Connector Bot
A Flask web application for analyzing video script connectors
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
from .script_connector_bot import ScriptConnectorBot
import PyPDF2

app = Flask(__name__)

# Initialize the bot
bot = ScriptConnectorBot()

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_script():
    """Analyze a script for connectors"""
    try:
        # Get script text and intro from request
        script_text = request.json.get('script_text', '')
        script_intro = request.json.get('script_intro', '')
        
        if not script_text:
            return jsonify({'error': 'No script text provided'}), 400
        
        # Initialize bot with API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            bot_with_ai = ScriptConnectorBot(openai_api_key=api_key)
        else:
            bot_with_ai = bot
        
        # Analyze the script with custom intro if provided
        analysis = bot_with_ai.parse_script(script_text, custom_intro=script_intro)
        
        # Convert analysis to JSON-serializable format
        result = {
            'score': analysis.score,
            'sections': [
                {
                    'number': section.number,
                    'title': section.title,
                    'content': section.content[:200] + '...' if len(section.content) > 200 else section.content,
                    'start_line': section.start_line,
                    'end_line': section.end_line
                }
                for section in analysis.sections
            ],
            'connectors': [
                {
                    'text': connector.text,
                    'type': connector.type.value,
                    'section_before': connector.section_before,
                    'section_after': connector.section_after,
                    'line_number': connector.line_number,
                    'is_valid': connector.is_valid,
                    'issues': connector.issues
                }
                for connector in analysis.connectors
            ],
            'missing_connectors': analysis.missing_connectors,
            'suggestions': analysis.suggestions,
            'intro': analysis.intro[:200] + '...' if len(analysis.intro) > 200 else analysis.intro,
            'payoff': analysis.payoff[:200] + '...' if len(analysis.payoff) > 200 else analysis.payoff
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/suggest_connector', methods=['POST'])
def suggest_connector():
    """Generate connector suggestions for a specific section transition"""
    try:
        data = request.json
        section_before_num = data.get('section_before')
        section_after_num = data.get('section_after')
        intro = data.get('intro', '')
        payoff = data.get('payoff', '')
        api_key = data.get('api_key', '')
        
        # Initialize bot with API key if provided
        if api_key:
            bot_with_ai = ScriptConnectorBot(openai_api_key=api_key)
        else:
            bot_with_ai = bot
        
        # Find the sections
        script_text = data.get('script_text', '')
        analysis = bot_with_ai.parse_script(script_text)
        
        section_before = next((s for s in analysis.sections if s.number == section_before_num), None)
        section_after = next((s for s in analysis.sections if s.number == section_after_num), None)
        
        if not section_before or not section_after:
            return jsonify({'error': 'Sections not found'}), 400
        
        # Generate suggestions
        suggestions = bot_with_ai.generate_connector_suggestions(section_before, section_after, intro, payoff)
        
        return jsonify({'suggestions': suggestions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Upload and extract text from PDF"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.lower().endswith('.pdf'):
            # Save the file temporarily
            file_path = f'temp_{file.filename}'
            file.save(file_path)
            
            # Extract text from PDF using pdfplumber for better formatting
            import pdfplumber
            text = ''
            
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
            
            # Clean up the file
            os.remove(file_path)
            
            # Clean up the text but preserve structure
            import re
            # Remove excessive whitespace but keep line breaks
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            return jsonify({'script_text': text})
        
        else:
            return jsonify({'error': 'Please upload a PDF file'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/segment_text', methods=['POST'])
def segment_text():
    """Segment text into logical sentences using AI"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        api_key = data.get('api_key')
        
        if not api_key:
            return jsonify({'error': 'OpenAI API key required'})
        
        # Initialize bot with API key
        bot = ScriptConnectorBot(openai_api_key=api_key)
        
        # Segment text
        segments = bot._segment_text_intelligently(text)
        
        return jsonify({'segments': segments})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
