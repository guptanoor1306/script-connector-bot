from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from script_connector_bot import ScriptConnectorBot
import pdfplumber

app = Flask(__name__, template_folder='templates', static_folder='../static')

# Initialize the bot
bot = ScriptConnectorBot()

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, '../static'),
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
                    'is_valid': connector.is_valid,
                    'line_number': connector.line_number,
                    'issues': connector.issues
                }
                for connector in analysis.connectors
            ],
            'suggestions': analysis.suggestions,
            'missing_connectors': [
                {'before': before, 'after': after}
                for before, after in analysis.missing_connectors
            ],
            'intro': analysis.intro,
            'payoff': analysis.payoff,
            'script_text': script_text
        }
        
        return jsonify(result)
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
        
        # Extract text using pdfplumber
        script_text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    script_text += text + "\n"
        
        # Clean up excessive whitespace
        script_text = '\n'.join(line.strip() for line in script_text.split('\n') if line.strip())
        
        return jsonify({'script_text': script_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
