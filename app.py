from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from transformers import LayoutLMForQuestionAnswering, LayoutLMTokenizer
import torch\

app = Flask(__name__)
CORS(app)

# Configuration for uploads and model
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

# Initialize the NLP model
tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMForQuestionAnswering.from_pretrained("microsoft/layoutlm-base-uncased")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        text = extract_text_from_pdf(filepath)

        # Save extracted text to a .txt file with UTF-8 encoding
        text_filename = os.path.splitext(filename)[0] + '.txt'
        text_filepath = os.path.join(app.config['UPLOAD_FOLDER'], text_filename)
        with open(text_filepath, 'w', encoding='utf-8') as text_file:
            text_file.write(text)

        return jsonify({"message": f"File {filename} uploaded successfully.", "text": text}), 200
    return jsonify({"error": "Invalid file type"}), 400

def extract_text_from_pdf(filepath):
    with open(filepath, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ''
    return text

@app.route('/answer-question', methods=['OPTIONS'])
def answer_question_options():
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    return response

@app.route('/answer-question', methods=['POST'])
def answer_question():
    data = request.get_json()
    resume_name = data.get('resume_name')
    question = data.get('question')

    # Load the resume text
    text_filename = resume_name + '.txt'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], text_filename)
    with open(filepath, 'r', encoding='utf-8') as file:
        resume_text = file.read()

    # Truncate the resume text to fit the model's max length
    max_length = 512  # Adjust this value based on your model's capabilities
    truncated_resume_text = resume_text[:max_length]

    inputs = tokenizer.encode_plus(question, truncated_resume_text, return_tensors="pt", truncation=True, max_length=max_length)

    outputs = model(**inputs)

    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))

    return jsonify({"answer": answer})

@app.route('/')
def index():
    return "Hello World"

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
