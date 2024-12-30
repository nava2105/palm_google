# palm_google

## Table of Contents
1. [General Info](#general-info)
2. [Technologies](#technologies)
3. [Installation](#installation)
4. [Usage](#usage)

## General Info
***
This project is a Python-based **PDF Management System** that allows users to:
- Upload and process PDF files to extract metadata and content.
- Generate embeddings using **Google Generative AI** to identify contract details such as provider, RUC, awarded value, and contract administrator if the uploaded file is an adjudication resolution, else if it is a star resolution, it extracts the members of the technical committee.
- Display extracted details in a GUI-based application using **Tkinter**.
- Edit and save JSON data containing metadata and responses.

The project leverages Google's **Gemini AI** model for content generation and embeddings, ensuring accurate data extraction.

## Technologies
***
A list of technologies used within the project:
- [Python](https://www.python.org): Version 3.12.0
- [Tkinter](https://docs.python.org/3/library/tkinter.html): GUI Framework
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/): Version 1.25.1 for PDF text extraction
- [PyPDF2](https://pypi.org/project/PyPDF2/): Version 3.0.1 for PDF metadata
- [LangChain](https://python.langchain.com/): Version 0.3.11 for text splitting
- [Google Generative AI](https://ai.google.dev/): Version 0.8.3 for embeddings and responses
- [NumPy](https://numpy.org/): Version 2.2.0 for cosine similarity
- [Tkinter](https://docs.python.org/3/library/tkinter.html): GUI Framework

## Installation
***
Follow these steps to set up the project:

### Prerequisites
Ensure you have Python 3.12.0 installed:
```bash
python --version
```

### Install the Project

#### Clone the Repository
```bash
git clone https://github.com/nava2105/palm_google.git
cd palm_google
```

#### Create and Activate a Virtual Environment
```bash
python -m venv .venv
# Activate on Windows
.venv\Scripts\activate
# Activate on macOS/Linux
source .venv/bin/activate
```

#### Install Required Dependencies
Install the required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the root directory and add your Google API key:
```plaintext
GOOGLE_API_KEY=your_google_api_key_here
```

### Run the Application
Run the main script:
```bash
python main.py
```

## Usage
***
### GUI Features
1. **Upload File:**
   - Upload PDF files to the system.
2. **Display Metadata and Content:**
   - View extracted PDF metadata and AI-processed details.
3. **Save JSON Data:**
   - Save extracted details into a JSON file.
4. **Edit JSON Data:**
   - Edit and save the JSON data directly from the interface.
5. **Download File:**
   - Download the uploaded PDF file.
6. **Search & Filter:**
   - Filter the list of uploaded files.

### Steps to Use
1. **Upload a PDF File:**
   - Use the **Upload File** button to select and upload a PDF.
2. **Process PDF:**
   - Select the file and click on **Details** to process and display extracted metadata and content.
3. **Save JSON:**
   - Click **Save JSON** to store the results in the `results` folders.
4. **Edit JSON:**
   - Use the **Edit JSON** feature to modify and save details.
5. **Download File:**
   - Download the uploaded PDF to your local machine.

### Outputs
- Extracted details are saved as JSON files in the `results` folder.
- Logs and errors are displayed for debugging.

### Folder Structure
```plaintext
palm_google/
├── main.py                      # Main application file
├── requirements.txt             # List of dependencies
├── .env                         # API key file
├── uploads_adjudication/        # Directory for uploaded adjudications resolutions
├── results_adjudication/        # Directory for JSON output of the adjudications resolutions
├── uploads_start/               # Directory for uploaded start resolutions
├── results_start/               # Directory for JSON output of the start resolutions
├── icon.ico                     # Icon of the application
├── loading.gif                  # Gif for the loading screen
└── README.md                    # Project documentation
```

## Notes
***
- **Disclaimer:** This application uses the Google Gemini API. Ensure you have a valid API key with appropriate billing and usage settings.
- The application will create folders `uploads` and `results` automatically if they do not exist.
