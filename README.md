# Resume Analyzer Web Application

This web application allows users to upload resumes in PDF format and ask questions about the content of the uploaded resumes. It uses Flask for the backend and LayoutLM model from Hugging Face for answering questions based on the resume content.

## Setup and Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package manager)
- A web browser
- uploads directory

### Installation Steps

1. **Clone the repository** (if applicable) or ensure all files are in your project directory.

2. **Create a virtual environment** (recommended):
   
   ```bash
   python -m venv venv

3. **Activate the virtual environment:**:

   ```bash
   venv\Scripts\activate

3. **Install dependencies:**:

   ```bash
   pip install flask PyPDF2 transformers torch flask-cors

4. **Run App:**:

   ```bash
   python app.py

4. **Open the HTML file:**:
   - Open the simple.html file in a web browser to interact with the application.
