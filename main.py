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
    Procesa una respuesta JSON y extrae campos específicos en formato de diccionario.
    """
    try:
        if not response.strip():
            return {"Error": "Response is empty or invalid."}

        data = json.loads(response)
        errors = []
        provider = data.get('proveedor') or "could not find the value"
        ruc = data.get('ruc') or "could not find the value"
        awarded_value = data.get('valor_adjudicado') or "could not find the value"
        administrator = data.get('administrador') or "could not find the value"

        # Construir el resultado como diccionario
        result = {
            "Provider": provider,
            "RUC": ruc,
            "Awarded Value": awarded_value,
            "Contract Administrator": administrator,
        }

        # Añadir errores si existen
        if "could not find the value" in [provider, ruc, awarded_value, administrator]:
            errors_list = []
            if provider == "could not find the value":
                errors_list.append("Provider: could not find the value")
            if ruc == "could not find the value":
                errors_list.append("RUC: could not find the value")
            if awarded_value == "could not find the value":
                errors_list.append("Awarded Value: could not find the value")
            if administrator == "could not find the value":
                errors_list.append("Contract Administrator: could not find the value")

            result["Errors"] = "\n".join(errors_list)

        return result

    except json.JSONDecodeError:
        return {"Error": "The response is not valid JSON."}


# Extract metadata from PDF
def parse_pdf_date(pdf_date):
    """Converts PDF metadata date to a readable format."""
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", pdf_date)
    if match:
        return f"{match[1]}-{match[2]}-{match[3]} {match[4]}:{match[5]}:{match[6]}"
    return "Fecha no disponible"

# Save details to JSON file
def save_to_json(metadata, response_data, filename):
    """
    Guarda los metadatos y datos de respuesta en un archivo JSON.
    """
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", f"{filename}.json")
    combined_data = {
        "Metadata": metadata,
        "Response": response_data
    }
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(combined_data, json_file, ensure_ascii=False, indent=4)
    print(f"Data saved to {output_path}")


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

        # Asociar evento para cargar detalles JSON automáticamente
        tree.bind("<<TreeviewSelect>>", load_json_details)

    def load_json_details(event):
        """
        Automatically loads JSON details of a selected file
        and displays them in a custom format.
        """
        selected_item = tree.selection()
        if selected_item:
            file_name = tree.item(selected_item, 'values')[1]
            base_name, _ = os.path.splitext(file_name)  # Quitar la extensión
            json_path = os.path.join("results", f"{base_name}.json")

            # Limpiar el widget de texto
            details_text.delete("1.0", tk.END)

            # Cargar y mostrar el JSON si existe
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)

                    # Formatear los metadatos y la respuesta
                    metadata = data.get("Metadata", {})
                    response = data.get("Response", {})

                    formatted_output = (
                        f"File Name: {metadata.get('filename', 'N/A')}\n"
                        f"Author: {metadata.get('author', 'N/A')}\n"
                        f"Created At: {metadata.get('created_at', 'N/A')}\n"
                        f"Modified At: {metadata.get('modified_at', 'N/A')}\n\n"
                        f"Provider: {response.get('Provider', 'N/A')}\n"
                        f"RUC: {response.get('RUC', 'N/A')}\n"
                        f"Awarded Value: {response.get('Awarded Value', 'N/A')}\n"
                        f"Contract Administrator: {response.get('Contract Administrator', 'N/A')}\n"
                    )

                    # Añadir errores si existen
                    if "Errors" in response:
                        formatted_output += f"Errors: {response.get('Errors')}\n"

                    # Mostrar el resultado formateado
                    details_text.insert(tk.END, formatted_output)
            else:
                details_text.insert(tk.END, f"No JSON file found for {file_name}.")

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
            process_pdf(file_path, file_name)

    def process_pdf(pdf_file_path, file_name):
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

        question = """To which provider was the contract awarded, what is the provider's RUC (consisting of 13 numerical values), what is the awarded value, and who is the contract administrator (make sure that this data is the administrator of the contract and that you are not confusing it with another person, if you are not sure, send this value empty)? The response must be given in the format:
        {proveedor: provider_name, ruc: provider_ruc, valor_adjudicado: awarded_value, administrador: contract_administrator}
        without any mor information or text than the one that is required and be sure to check at least 2 times the data provided before submitting your response."""
        response = generate_response_from_embeddings(question, embeddings_with_chunks)
        result = extract_data_from_response(response)
        # Format 'result' to be readable if it is a dictionary
        if isinstance(result, dict):
            formatted_result = "\n".join([f"{key}: {value}" for key, value in result.items()])
        else:
            formatted_result = result

        details_text.insert(tk.END, formatted_result)

        print("metadata:", metadata)
        print("response:", response)
        print("result:", result)
        # Save to JSON
        save_to_json(metadata, result, os.path.splitext(file_name)[0])

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
