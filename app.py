from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
import sys
import os

# Add the directory containing your query_data.py to the Python path
# This ensures Flask can find your custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from query_data import query_rag # Import your query_rag function

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Call your query_rag function
        # query_rag prints directly, but we need its return value for the API
        # Make sure query_rag returns the response_text and sources
        # Current query_data.py returns response_text, we'll assume it's good for now.
        # If you want sources, you'll need to modify query_data.py's return.
        full_response = query_rag(question) # Assuming query_rag returns the full formatted string
        
        # Parse the response to separate content and sources if query_rag returns a single string like "Response: ... Sources: [...]"
        response_lines = full_response.split('\nSources: ')
        response_text = response_lines[0].replace('Response: ', '').strip()
        sources = []
        if len(response_lines) > 1:
            try:
                sources_str = response_lines[1].strip()
                # Safely evaluate the string representation of the list
                sources = eval(sources_str) 
            except Exception as e:
                print(f"Could not parse sources: {sources_str} Error: {e}")
                sources = ["Error parsing sources"]


        return jsonify({
            'answer': response_text,
            'sources': sources
        })
    except Exception as e:
        # Log the full exception for debugging
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    # Make sure your database is populated before running the Flask app
    # C:\Python312\python.exe D:\rag-tutorial-v2\populate_database.py --reset
    app.run(debug=True, port=5000) # Run on port 5000
