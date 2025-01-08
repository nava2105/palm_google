# utilities.py
import os
import shutil
import socket
from tkinter import messagebox


class Utilities:
    """
    A class housing utility methods for file manipulation, directory creation, and internet connection verification.
    """

    @staticmethod
    def ensure_folders_exist(folders):
        """
        Ensure that required folders exist by creating them if they don't already exist.
        """
        try:
            for folder in folders:
                os.makedirs(folder, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create required folders: {e}")
            raise

    @staticmethod
    def verify_internet_connection(host="8.8.8.8", port=53, timeout=3):
        """
        Verify if the computer has an active internet connection.
        """
        try:
            socket.create_connection((host, port), timeout=timeout)
            return True
        except (socket.timeout, OSError):
            # No connection available; show a warning
            messagebox.showwarning(
                "No Internet Connection",
                (
                    "This app requires an active internet connection to use Gemini AI functionalities. "
                    "You can still review or modify saved data but cannot request new data from the AI. "
                    "Make sure you connect to the internet for full functionality."
                )
            )
            return False

    @staticmethod
    def get_file_path(file_name, document_type):
        """
        Generate the file path based on the document type.
        """
        folder = "uploads_adjudication" if document_type == "Adjudications Resolution" else "uploads_start"
        return os.path.join(folder, file_name)

    @staticmethod
    def get_json_path(base_name, document_type):
        """
        Generate the JSON file path for a specific document type.
        """
        folder = "results_adjudication" if document_type == "Adjudications Resolution" else "results_start"
        return os.path.join(folder, f"{base_name}.json")

    @staticmethod
    def copy_to_folder(file_path, destination_folder):
        """
        Copy a file to the specified folder.
        """
        try:
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            shutil.copy(file_path, destination_folder)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy file: {file_path}\n{e}")
            raise