# text_service.py

import json
import os
from tkinter import messagebox


class TextService:
    def __init__(self):
        """
        TextService class constructor.
        """
        pass

    @staticmethod
    def extract_data_from_response(response, document_type):
        """
        It processes a response in JSON format and extracts specific fields based on the document type.
        """
        try:
            if not response.strip():
                return {"Error": "Response is empty or invalid."}

            data = json.loads(response)

            if document_type == "Adjudications Resolution":
                provider = data.get("proveedor", "could not find the value")
                ruc = data.get("ruc", "could not find the value")
                awarded_value = data.get("valor_adjudicado", "could not find the value")
                administrator = data.get("administrador", "could not find the value")

                result = {
                    "Provider": provider,
                    "RUC": ruc,
                    "Awarded Value": awarded_value,
                    "Contract Administrator": administrator,
                }
                return result

            elif document_type == "Start Resolution":
                formulated_the_requirement = data.get("Formulated the Requirement", "could not find the value")
                approved_the_requirement = data.get("Approved the Requirement", "could not find the value")
                delegate_of_the_highest_authority = data.get("Delegate of the Highest Authority",
                                                             "could not find the value")
                contract_administrator = data.get("Contract Administrator", "could not find the value")

                result = {
                    "Formulated the Requirement": formulated_the_requirement,
                    "Approved the Requirement": approved_the_requirement,
                    "Delegate of the Highest Authority": delegate_of_the_highest_authority,
                    "Contract Administrator": contract_administrator,
                }
                return result

        except json.JSONDecodeError:
            return {"Error": "The response is not valid JSON."}

    @staticmethod
    def save_to_json(metadata, response_data, filename, document_type):
        """
        Store the extracted metadata and data in a JSON file separated by document type.
        """
        # Create output directory based on document type
        if document_type == "Adjudications Resolution":
            output_path = os.path.join("results_adjudication", f"{filename}.json")
        else:
            output_path = os.path.join("results_start", f"{filename}.json")

        # Combine metadata and response
        combined_data = {
            "Metadata": metadata,
            "Response": response_data
        }

        # Save in JSON format
        try:
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(combined_data, json_file, ensure_ascii=False, indent=4)
            print(f"Data saved to {output_path}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")
            messagebox.showerror("Error", f"Failed to save JSON file: {str(e)}")

    @staticmethod
    def parse_formatted_text_to_save(edited_content, document_type):
        """
        Converts the content of the text widget (JSON details) into a dictionary for storage.
        """
        try:
            lines = edited_content.split("\n")
            metadata = {
                "filename": lines[0].split(": ", 1)[1],
                "author": lines[1].split(": ", 1)[1],
                "created_at": lines[2].split(": ", 1)[1],
                "modified_at": lines[3].split(": ", 1)[1],
            }

            if document_type == "Adjudications Resolution":
                response = {
                    "Provider": lines[5].split(": ", 1)[1],
                    "RUC": lines[6].split(": ", 1)[1],
                    "Awarded Value": lines[7].split(": ", 1)[1],
                    "Contract Administrator": lines[8].split(": ", 1)[1],
                }
            else:  # Start Resolution
                response = {
                    "Formulated the Requirement": lines[5].split(": ", 1)[1],
                    "Approved the Requirement": lines[6].split(": ", 1)[1],
                    "Delegate of the Highest Authority": lines[7].split(": ", 1)[1],
                    "Contract Administrator": lines[8].split(": ", 1)[1],
                }

            return {"Metadata": metadata, "Response": response}
        except Exception as e:
            print(f"Error parsing edited text: {e}")
            return {"Error": "Failed to parse the edited content. Please check the format."}

    @staticmethod
    def save_parsed_json_to_file(parsed_data, json_file_path):
        """
        Saves the processed data (metadata and response) in a JSON file.
        """
        try:
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(parsed_data, json_file, ensure_ascii=False, indent=4)
            messagebox.showinfo("Success", f"JSON file saved successfully: {json_file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save JSON file: {e}")
            print(f"Error saving JSON file: {e}")