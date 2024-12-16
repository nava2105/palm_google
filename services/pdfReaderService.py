from PyPDF2 import PdfReader
import os
import re

class PDFReaderService:
    @staticmethod
    def parse_pdf_date(pdf_date):
        """Convierte la fecha de metadatos de PDF a un formato legible."""
        match = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", pdf_date)
        if match:
            return f"{match[1]}-{match[2]}-{match[3]} {match[4]}:{match[5]}:{match[6]}"
        return "Fecha no disponible"

    @staticmethod
    def extract_metadata(filepath):
        """Extrae metadatos del PDF en la ruta proporcionada."""
        details = {"filename": os.path.basename(filepath), "author": "No disponible", "created_at": "N/A", "modified_at": "N/A"}
        try:
            reader = PdfReader(filepath)  # Abre directamente la ruta recibida
            metadata = reader.metadata
            if metadata:
                details["author"] = metadata.get('/Author', "No disponible")
                details["created_at"] = PDFReaderService.parse_pdf_date(metadata.get('/CreationDate', ""))
                details["modified_at"] = PDFReaderService.parse_pdf_date(metadata.get('/ModDate', ""))
        except Exception as e:
            print(f"Error al leer PDF: {e}")
        return details

    @staticmethod
    def extract_text(filepath):
        """Extrae el texto completo de un archivo PDF."""
        text = ""
        try:
            reader = PdfReader(filepath)  # Abre directamente la ruta recibida
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print(f"Error al extraer texto del PDF: {e}")
        return text
