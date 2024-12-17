from PyPDF2 import PdfReader
import re
import os
import textwrap
import google.generativeai as genai
import fitz
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import shutil

# Configure the API Key
def configure_api():
    """
    Configures the API key for Google Generative AI.
    """
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)

# Format text to Markdown
def format_markdown(text):
    """
    Formats text into Markdown style.
    """
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

# Read PDF content
def read_pdf_content(file_path):
    """
    Reads and extracts text content from a PDF file.
    """
    try:
        content = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                content += page.get_text()
        return content
    except Exception as error:
        messagebox.showerror("Error", f"Error reading PDF file: {error}")
        return ""

# Generate content based on embeddings
def generate_response_from_embeddings(question, embeddings_with_chunks, model_name='gemini-1.5-flash'):
    """
    Generates a response using the question and relevant embeddings.
    """
    model = genai.GenerativeModel(model_name)

    # Generate embedding for the question
    question_embedding = genai.embed_content(
        model="models/embedding-001",
        content=question,
        task_type="retrieval_document",
        title="Question"
    )['embedding']

    def calculate_cosine_similarity(vector_a, vector_b):
        return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

    # Find the most relevant chunk
    similarities = [
        (calculate_cosine_similarity(question_embedding, embedding), chunk)
        for chunk, embedding in embeddings_with_chunks
    ]
    most_relevant_chunk = max(similarities, key=lambda x: x[0])[1]

    # Generate the response using the most relevant chunk
    prompt = f"Context:\n{most_relevant_chunk}\n\nQuestion:\n{question}"
    response = model.generate_content(prompt)

    # Clean JSON formatting
    cleaned_response = response.text.replace("```json", "").replace("```", "").strip()
    return cleaned_response

# Split text into chunks
def split_text_into_chunks(text):
    """
    Splits the input text into manageable chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=8800,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Generate embeddings for a given text
def generate_text_embedding(text, title="Text Chunk"):
    """
    Generates embeddings for the provided text using Google's embedding model.
    """
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
        title=title
    )
    return result['embedding']

# Process JSON response and extract data
def extract_data_from_response(response):
    """
    Processes a JSON-formatted response and extracts specific fields.
    Throws an error for fields with null or missing values.
    """
    print(response)
    try:
        if not response.strip():
            return "Error: Response is empty or invalid."

        data = json.loads(response)
        errors = []
        provider = data.get('proveedor')
        ruc = data.get('ruc')
        awarded_value = data.get('valor_adjudicado')
        administrator = data.get('administrador')

        # Check for null or missing fields
        if not provider:
            errors.append("Provider: could not find the value")
        if not ruc:
            errors.append("RUC: could not find the value")
        if not awarded_value:
            errors.append("Awarded Value: could not find the value")
        if not administrator:
            errors.append("Contract Administrator: could not find the value")

        # Format result with errors or valid values
        result = [
            f"Provider: {provider or 'could not find the value'}",
            f"RUC: {ruc or 'could not find the value'}",
            f"Awarded Value: {awarded_value or 'could not find the value'}",
            f"Contract Administrator: {administrator or 'could not find the value'}"
        ]

        if errors:
            result.append("\nErrors:\n" + "\n".join(errors))

        return "\n".join(result)

    except json.JSONDecodeError:
        return "Error: The response is not valid JSON."

# Extract metadata from PDF
def parse_pdf_date(pdf_date):
    """Converts PDF metadata date to a readable format."""
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", pdf_date)
    if match:
        return f"{match[1]}-{match[2]}-{match[3]} {match[4]}:{match[5]}:{match[6]}"
    return "Fecha no disponible"

def extract_metadata(filepath):
    """
    Extracts metadata from a PDF file.
    """
    details = {"filename": os.path.basename(filepath), "author": "No disponible", "created_at": "N/A", "modified_at": "N/A"}
    try:
        reader = PdfReader(filepath)
        metadata = reader.metadata
        if metadata:
            details["author"] = metadata.get('/Author', "No disponible")
            details["created_at"] = parse_pdf_date(metadata.get('/CreationDate', ""))
            details["modified_at"] = parse_pdf_date(metadata.get('/ModDate', ""))
    except Exception as e:
        print(f"Error al leer PDF: {e}")
    return details

def extract_text(filepath):
    """
    Extracts full text content from a PDF file.
    """
    text = ""
    try:
        reader = PdfReader(filepath)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error al extraer texto del PDF: {e}")
    return text

# GUI Application to process and manage files
def main_gui():
    def upload_file():
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            os.makedirs("uploads", exist_ok=True)
            new_file_path = os.path.join("uploads", os.path.basename(file_path))
            shutil.copy(file_path, new_file_path)
            load_file_table()

    def load_file_table():
        for row in tree.get_children():
            tree.delete(row)
        files = os.listdir("uploads")
        for index, file in enumerate(files):
            tree.insert("", "end", values=(index + 1, file))

    def download_file():
        selected_item = tree.selection()
        if selected_item:
            file_name = tree.item(selected_item, 'values')[1]
            file_path = os.path.join("uploads", file_name)
            destination = filedialog.askdirectory()
            if destination:
                shutil.copy(file_path, os.path.join(destination, file_name))
                messagebox.showinfo("Success", f"File {file_name} downloaded successfully!")

    def show_details():
        selected_item = tree.selection()
        if selected_item:
            file_name = tree.item(selected_item, 'values')[1]
            file_path = os.path.join("uploads", file_name)
            process_pdf(file_path)

    def process_pdf(pdf_file_path):
        metadata = extract_metadata(pdf_file_path)
        details_text.delete("1.0", tk.END)
        details_text.insert(tk.END, f"File Name: {metadata['filename']}\n")
        details_text.insert(tk.END, f"Author: {metadata['author']}\n")
        details_text.insert(tk.END, f"Created At: {metadata['created_at']}\n")
        details_text.insert(tk.END, f"Modified At: {metadata['modified_at']}\n\n")

        configure_api()
        pdf_content = read_pdf_content(pdf_file_path)
        if not pdf_content:
            return

        chunks = split_text_into_chunks(pdf_content)
        embeddings_with_chunks = [(chunk, generate_text_embedding(chunk)) for chunk in chunks]

        question = "To which provider was the contract awarded, what is the provider's RUC (consisting of 13 numerical values), what is the awarded value, and who is the contract administrator (make sure that this data is the administrator of the contract and that you are not confusing it with another person, if you are not sure, send this value empty)? The response must be given in the format:\n{\nproveedor: provider_name,\nruc: provider_ruc,\nvalor_adjudicado: awarded_value,\nadministrador: contract_administrator\n}\nwithout any mor information or text than the one that is required and be sure to check at least 2 times the data provided before submitting your response."
        response = generate_response_from_embeddings(question, embeddings_with_chunks)
        result = extract_data_from_response(response)
        details_text.insert(tk.END, result)

    root = tk.Tk()
    root.title("PDF Management System")
    root.state('zoomed')

    # Title of the application
    title_label = tk.Label(root, text="PDF Management System", font=("Helvetica", 18, "bold"))
    title_label.pack(pady=10)

    # Menu with buttons
    menu_frame = tk.Frame(root)
    menu_frame.pack(fill="x", pady=5, padx=50)

    tk.Button(menu_frame, text="Upload File", command=upload_file).pack(side="left", padx=5, pady=5)
    tk.Button(menu_frame, text="Download File", command=download_file).pack(side="left", padx=5, pady=5)
    tk.Button(menu_frame, text="Details", command=show_details).pack(side="left", padx=5, pady=5)

    # Main Content Frame
    content_frame = tk.Frame(root)
    content_frame.pack(expand=True, fill="both", padx=30, pady=30)

    # Left Frame (File Table)
    left_frame = tk.Frame(content_frame)
    left_frame.grid(row=0, column=0, sticky="nsew", padx=15)

    tk.Label(left_frame, text="Uploaded Files:", font=("Helvetica", 12, "bold")).pack(pady=5)
    columns = ("#", "File Name")
    global tree
    tree = ttk.Treeview(left_frame, columns=columns, show="headings")

    tree.column("#", width=30, anchor="center")
    tree.heading("#", text="#")

    tree.column("File Name", anchor="w", stretch=True)
    tree.heading("File Name", text="File Name", anchor="w")

    tree.pack(expand=True, fill="both")
    load_file_table()

    # Right Frame (Details Text Output)
    right_frame = tk.Frame(content_frame)
    right_frame.grid(row=0, column=1, sticky="nsew", padx=15)

    tk.Label(right_frame, text="File Details:", font=("Helvetica", 12, "bold")).pack(pady=5)
    global details_text
    details_text = tk.Text(right_frame, wrap="word")
    details_text.pack(expand=True, fill="both")

    # Configure grid layout for equal column width
    content_frame.grid_columnconfigure(0, weight=1)
    content_frame.grid_columnconfigure(1, weight=1)
    content_frame.grid_rowconfigure(0, weight=1)

    root.mainloop()


if __name__ == "__main__":
    main_gui()
