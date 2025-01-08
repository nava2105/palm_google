# pdf_service.py

import os
import re
from tkinter import messagebox
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter


class PDFService:
    def __init__(self):
        """
        PDFService class constructor.
        """
        pass

    @staticmethod
    def read_pdf_content(file_path):
        """
        Reads and extracts the textual content of a PDF file.
        """
        try:
            pdf_text = ""
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    pdf_text += page.get_text()
            return pdf_text
        except Exception as error:
            messagebox.showerror("Error", f"Error reading PDF file: {error}")
            return ""

    @staticmethod
    def split_text_into_chunks(text, chunk_size=7000, overlap=2000):
        """
        Split the text into manageable chunks based on the given size.
        """
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len
        )
        return text_splitter.split_text(text)

    @staticmethod
    def parse_pdf_date(pdf_date):
        """
        Parses and formats a date from PDF metadata into a readable format.
        """
        match = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", pdf_date)
        if match:
            return f"{match[1]}-{match[2]}-{match[3]} {match[4]}:{match[5]}:{match[6]}"
        return "Date not available"

    @staticmethod
    def extract_metadata(file_path):
        """
        Extracts the main metadata of a PDF file, such as author, creation date and modification date.
        """
        metadata_details = {
            "filename": os.path.basename(file_path),
            "author": "Not available",
            "created_at": "Not available",
            "modified_at": "Not available"
        }

        try:
            reader = PdfReader(file_path)
            metadata = reader.metadata

            if metadata:
                # Extraer y validar campos de metadatos
                author = metadata.get('/Author')
                created_date = metadata.get('/CreationDate')
                modified_date = metadata.get('/ModDate')

                metadata_details["author"] = author if isinstance(author, str) else "Not available"
                metadata_details["created_at"] = (
                    PDFService.parse_pdf_date(created_date) if isinstance(created_date, str) else "Not available"
                )
                metadata_details["modified_at"] = (
                    PDFService.parse_pdf_date(modified_date) if isinstance(modified_date, str) else "Not available"
                )
        except Exception as error:
            print(f"Error reading PDF metadata: {error}")

        return metadata_details