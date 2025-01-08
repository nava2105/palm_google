# ---Imports---
# Standard library imports
import os
import json
import shutil
import threading
from dotenv import load_dotenv

# Third-party library imports
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Classes
from api_service import APIService
from pdf_service import PDFService
from text_service import TextService
from utilities import Utilities

# ---Initialize---
# Load the environment variables
load_dotenv()

# Create instances
api_service = APIService()
pdf_service = PDFService()
text_service = TextService()
utilities = Utilities()


# ---UserInterface---
# GUI Application to process and manage files
def main_gui():
    # ---GUIMethods---
    def upload_file():
        """
        Upload a file to the folder.
        """
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            selected_type = document_type.get()
            destination_folder = "uploads_adjudication" if selected_type == "Adjudications Resolution" else "uploads_start"
            utilities.copy_to_folder(file_path, destination_folder)
            load_file_table()

    def get_json_path(selected_item, tree, document_type):
        """
        Generates the JSON path based on the selected item and document type.
        """
        file_name = tree.item(selected_item, 'values')[1]
        base_name, _ = os.path.splitext(file_name)
        return utilities.get_json_path(base_name, document_type)

    def load_json_details(event):
        """
        Automatically loads JSON details of a selected file and displays them in a custom format.
        """
        global details_text
        if details_text is None:
            return

        selected_item = tree.selection()
        if selected_item:
            file_name = tree.item(selected_item, 'values')[1]
            base_name, _ = os.path.splitext(file_name)
            selected_type = document_type.get()
            json_path = get_json_path(selected_item, tree, document_type.get())

            details_text.config(state="normal")
            details_text.delete("1.0", tk.END)

            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    metadata = data.get("Metadata", {})
                    response = data.get("Response", {})

                    selected_type = document_type.get()
                    if selected_type == "Adjudications Resolution":
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
                    else:
                        formatted_output = (
                            f"File Name: {metadata.get('filename', 'N/A')}\n"
                            f"Author: {metadata.get('author', 'N/A')}\n"
                            f"Created At: {metadata.get('created_at', 'N/A')}\n"
                            f"Modified At: {metadata.get('modified_at', 'N/A')}\n\n"
                            f"Formulated the Requirement: {response.get('Formulated the Requirement', 'N/A')}\n"
                            f"Approved the Requirement: {response.get('Approved the Requirement', 'N/A')}\n"
                            f"Delegate of the Highest Authority: {response.get('Delegate of the Highest Authority', 'N/A')}\n"
                            f"Contract Administrator: {response.get('Contract Administrator', 'N/A')}\n"
                        )

                    details_text.insert(tk.END, formatted_output)
            else:
                details_text.insert(tk.END, f"No JSON file found for {file_name}.")

            details_text.config(state="disabled")

    def download_file():
        """
        Downloads the selected file.
        """
        selected_item = tree.selection()
        if selected_item:
            file_name = tree.item(selected_item, 'values')[1]
            selected_type = document_type.get()
            if selected_type == "Adjudications Resolution":
                file_path = os.path.join("uploads_adjudication", file_name)
            else:
                file_path = os.path.join("uploads_start", file_name)

            destination = filedialog.askdirectory()
            if destination:
                shutil.copy(file_path, os.path.join(destination, file_name))
                messagebox.showinfo("Success", f"File {file_name} downloaded successfully!")

    def show_details():
        """
        Shows the details of the selected file with a fullscreen semi-transparent loading screen displaying an animated GIF.
        """
        user_response = messagebox.askyesno(
            "See details",
            "You will ask Gemini IA to search for the information inside the PDF. Every change made manually is going to be lost! Do you want to proceed?"
        )

        if user_response:  # Proceed only if the user clicks 'Yes'
            selected_item = tree.selection()
            if selected_item:
                file_name = tree.item(selected_item, 'values')[1]

                selected_type = document_type.get()
                if selected_type == "Adjudications Resolution":
                    file_path = os.path.join("uploads_adjudication", file_name)
                else:
                    file_path = os.path.join("uploads_start", file_name)

                # Create a fullscreen semi-transparent loading screen
                loading_screen = tk.Toplevel(root)
                loading_screen.attributes("-fullscreen", True)  # Make it fullscreen
                loading_screen.attributes("-alpha", 0.8)  # Set transparency
                loading_screen.configure(bg="black")  # Background color

                # Load the GIF
                gif_path = "loading.gif"  # Path to your loading GIF
                gif_image = Image.open(gif_path)
                gif_frames = []
                try:
                    while True:
                        gif_frames.append(ImageTk.PhotoImage(gif_image.copy().convert("RGBA")))
                        gif_image.seek(len(gif_frames))  # Move to the next frame
                except EOFError:
                    pass  # End of GIF frames

                # Create a label to display the GIF
                gif_label = tk.Label(loading_screen, bg="black")
                gif_label.pack(expand=True)

                def animate_gif(index=0):
                    """Animate the GIF frame by frame."""
                    gif_label.configure(image=gif_frames[index])
                    next_index = (index + 1) % len(gif_frames)
                    loading_screen.after(100, animate_gif, next_index)  # Adjust delay (100ms) based on your GIF's speed

                # Start the animation
                animate_gif()

                # Add a message below the GIF
                message_label = tk.Label(
                    loading_screen,
                    text="Processing, please wait...",
                    font=("Helvetica", 24),
                    bg="black",
                    fg="white"
                )
                message_label.pack(pady=20)

                loading_screen.transient(root)  # Keep it on top of the main window
                loading_screen.grab_set()  # Disable interactions with the main window

                def process_task():
                    """Run the PDF processing in a separate thread."""
                    try:
                        process_pdf(file_path, file_name)
                    finally:
                        loading_screen.destroy()  # Close the loading screen once done

                # Run the processing in a separate thread
                threading.Thread(target=process_task, daemon=True).start()
        else:
            messagebox.showinfo("Action Cancelled", "No changes were made.")

    def process_pdf(pdf_file_path, file_name):
        """"
        Processes the selected file and extracts it's metadata and the response given by gemini.
        """
        metadata = pdf_service.extract_metadata(pdf_file_path)
        details_text.delete("1.0", tk.END)
        details_text.insert(tk.END, f"File Name: {metadata['filename']}\n")
        details_text.insert(tk.END, f"Author: {metadata['author']}\n")
        details_text.insert(tk.END, f"Created At: {metadata['created_at']}\n")
        details_text.insert(tk.END, f"Modified At: {metadata['modified_at']}\n\n")

        if not utilities.verify_internet_connection():
            # Stop further processing if there's no internet connection
            return

        api_service.configure_api()
        pdf_content = pdf_service.read_pdf_content(pdf_file_path)
        if not pdf_content:
            return

        chunks = pdf_service.split_text_into_chunks(pdf_content)
        embeddings_with_chunks = [(chunk, api_service.generate_text_embedding(chunk)) for chunk in chunks]

        # Choose prompt based on document type
        selected_type = document_type.get()
        if selected_type == "Adjudications Resolution":
            question = """To which provider was the contract awarded, what is the provider's RUC (consisting of 13 numerical values), what is the awarded value, and who is the contract administrator (make sure that this data is the administrator of the contract and that you are not confusing it with another person, if you are not sure, send this value empty)? The response must be given in the format:
            {proveedor: provider_name, ruc: provider_ruc, valor_adjudicado: awarded_value, administrador: contract_administrator}
            without any mor information or text than the one that is required and be sure to check at least 2 times the data provided before submitting your response."""
        else:
            question = """If you do not find them, provide the available details for the following roles:
            - Who formulated the requirement.
            - Who approved the requirement.
            - The delegate of the highest authority.
            - The contract administrator.
            If any information is missing, include the key with an empty value. Format the response in JSON as follows:
            {
              "Formulated the Requirement": "Full Name, Office Held",
              "Approved the Requirement": "Full Name, Office Held",
              "Delegate of the Highest Authority": "Full Name, Office Held",
              "Contract Administrator": "Full Name, Office Held"
            }"""
        response = api_service.generate_response_from_embeddings(question, embeddings_with_chunks)
        result = text_service.extract_data_from_response(response, selected_type)
        # Format 'result' to be readable if it is a dictionary
        if isinstance(result, dict):
            formatted_result = "\n".join([f"{key}: {value}" for key, value in result.items()])
        else:
            formatted_result = result

        details_text.insert(tk.END, formatted_result)

        text_service.save_to_json(metadata, result, os.path.splitext(file_name)[0], selected_type)

    def enable_json_editing():
        """
        Enables editing of the text in the detail's widget.
        """
        details_text.config(state="normal")  # Allows you to edit the text widget
        messagebox.showinfo("Edit Mode", "You can now edit the JSON content. Don't forget to save your changes!")

    def refresh_application():
        """
        Refresh the application interface:
        - Restart Treeview.
        - Clean and update File Details.
        """
        load_file_table()

        # Clean the detail area
        details_text.delete("1.0", tk.END)
        details_text.config(state="disabled")

    def save_json_edits():
        """
        Saves edits made manually in the text widget back to the JSON file.
        """
        def save_json_task():
            """"
            Saves the json with the data
            """
            selected_item = tree.selection()
            if selected_item:
                file_name = tree.item(selected_item, 'values')[1]
                base_name, _ = os.path.splitext(file_name)
                selected_type = document_type.get()
                json_path = get_json_path(selected_item, tree, document_type.get())

                try:
                    edited_content = details_text.get("1.0", tk.END).strip()

                    lines = edited_content.split("\n")
                    metadata = {
                        "filename": lines[0].split(": ", 1)[1],
                        "author": lines[1].split(": ", 1)[1],
                        "created_at": lines[2].split(": ", 1)[1],
                        "modified_at": lines[3].split(": ", 1)[1],
                    }
                    selected_type = document_type.get()
                    if selected_type == "Adjudications Resolution":
                        response = {
                            "Provider": lines[5].split(": ", 1)[1],
                            "RUC": lines[6].split(": ", 1)[1],
                            "Awarded Value": lines[7].split(": ", 1)[1],
                            "Contract Administrator": lines[8].split(": ", 1)[1],
                        }
                    else:
                        response = {
                            "Formulated the Requirement": lines[5].split(": ", 1)[1],
                            "Approved the Requirement": lines[6].split(": ", 1)[1],
                            "Delegate of the Highest Authority": lines[7].split(": ", 1)[1],
                            "Contract Administrator": lines[8].split(": ", 1)[1],
                        }

                    updated_data = {"Metadata": metadata, "Response": response}

                    with open(json_path, 'w', encoding='utf-8') as json_file:
                        json.dump(updated_data, json_file, ensure_ascii=False, indent=4)

                    root.after(0, lambda: messagebox.showinfo("Success", f"JSON file saved successfully: {json_path}"))
                    root.after(0, refresh_application)

                except Exception as e:
                    root.after(0, lambda: messagebox.showerror("Error", f"Failed to save JSON file: {str(e)}"))

        # Runs the task on a secondary thread
        threading.Thread(target=save_json_task).start()

    def load_file_table(filter_text=""):
        """
        Load the list of files into Treeview and apply optional filters.
        """
        global tree
        for widget in left_frame.winfo_children():
            widget.destroy()

        scrollbar = ttk.Scrollbar(left_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        columns = ("#", "File Name")
        tree = ttk.Treeview(left_frame, columns=columns, show="headings", yscrollcommand=scrollbar.set)

        tree.column("#", width=30, anchor="center")
        tree.heading("#", text="#")

        tree.column("File Name", anchor="w", stretch=True)
        tree.heading("File Name", text="File Name", anchor="w")

        scrollbar.config(command=tree.yview)
        tree.pack(expand=True, fill="both")
        selected_type = document_type.get()

        if selected_type == "Adjudications Resolution":
            files = os.listdir("uploads_adjudication")
        else:
            files = os.listdir("uploads_start")

        # Filter files according to the text entered
        filtered_files = [file for file in files if filter_text.lower() in file.lower()]

        for index, file in enumerate(filtered_files):
            tree.insert("", "end", values=(index + 1, file))

        tree.bind("<<TreeviewSelect>>", load_json_details)

        # Initial selection if there are files
        if filtered_files:
            first_item = tree.get_children()[0]
            tree.selection_set(first_item)
            tree.event_generate("<<TreeviewSelect>>")

    def filter_files():
        """
        Filter files based on the input text from the search bar.
        """
        filter_text = search_entry.get()
        load_file_table(filter_text)

    root = tk.Tk()
    root.title("PDF Management System")
    root.state('zoomed')
    root.iconbitmap("icon.ico")

    # Colour palette
    BG_COLOR = "#53a2be"  # General Background
    FG_COLOR = "#fdfdff"  # Main text colour
    BUTTON_BG = "#fdfdff"  # Button background
    BUTTON_FG = "#003554"  # Button text colour
    SELECTED_BG = "#575757"  # Background when selecting a file

    global details_text
    details_text = None

    # Global window configuration
    root.configure(bg=BG_COLOR)

    # Title of the application
    title_label = tk.Label(root, text="PDF Management System", font=("Helvetica", 32, "bold"), bg=BG_COLOR, fg=FG_COLOR)
    title_label.pack(pady=10)

    # Menu with buttons
    menu_frame = tk.Frame(root, bg=BG_COLOR)
    menu_frame.pack(fill="x", pady=5, padx=50)

    button_style = {"bg": BUTTON_BG, "fg": BUTTON_FG, "activebackground": SELECTED_BG}
    search_button = tk.Button(menu_frame, text="Filter", command=filter_files, **button_style)
    search_button.pack(side="right", padx=5, pady=5)

    search_entry = tk.Entry(menu_frame, width=50)
    search_entry.pack(side="right", padx=5, pady=5)
    search_entry.bind("<Return>", lambda event: filter_files())

    search_label = tk.Label(menu_frame, text="Search:", bg=BG_COLOR, fg=FG_COLOR, font=("Helvetica", 10))
    search_label.pack(side="right", padx=5, pady=5)

    document_type = tk.StringVar()
    document_type_dropdown = ttk.Combobox(
        menu_frame, textvariable=document_type, values=["Adjudications Resolution", "Start Resolution"],
    )
    document_type_label = tk.Label(menu_frame, text="Document Type:", bg=BG_COLOR, fg=FG_COLOR)
    document_type_label.pack(side="left", padx=5)

    document_type_dropdown.pack(side="left", padx=5)
    document_type_dropdown.current(0)  # Default selection

    document_type.trace_add("write", lambda *args: load_file_table(filter_text=""))  # Automatically refresh the file table with an empty filter

    tk.Label(menu_frame, bg=BG_COLOR).pack(side="left", padx=50)

    tk.Button(menu_frame, text="Upload File", command=upload_file, **button_style).pack(side="left", padx=5, pady=5)
    tk.Button(menu_frame, text="Download File", command=download_file, **button_style).pack(side="left", padx=5, pady=5)
    tk.Button(menu_frame, text="Details", command=show_details, **button_style).pack(side="left", padx=5, pady=5)
    tk.Button(menu_frame, text="Edit JSON", command=enable_json_editing, **button_style).pack(side="left", padx=5, pady=5)
    tk.Button(menu_frame, text="Save JSON", command=save_json_edits, **button_style).pack(side="left", padx=5, pady=5)

    # Main Content Frame
    content_frame = tk.Frame(root, bg=BG_COLOR)
    content_frame.pack(expand=True, fill="both", padx=30, pady=30)

    # Left Frame (File Table)
    left_frame = tk.Frame(content_frame, bg=BG_COLOR)
    left_frame.grid(row=0, column=0, sticky="nsew", padx=15)

    tk.Label(left_frame, text="Uploaded Files:", font=("Helvetica", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(pady=5)
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
    right_frame = tk.Frame(content_frame, bg=BG_COLOR)
    right_frame.grid(row=0, column=1, sticky="nsew", padx=15)

    tk.Label(right_frame, text="File Details:", font=("Helvetica", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(pady=5)

    details_text = tk.Text(right_frame, wrap="word")
    details_text.pack(expand=True, fill="both")

    # Configure grid layout for equal column width
    content_frame.grid_columnconfigure(0, weight=1)
    content_frame.grid_columnconfigure(1, weight=1)
    content_frame.grid_rowconfigure(0, weight=1)

    # Footer Frame (Disclaimer)
    footer_frame = tk.Frame(root, bg=BG_COLOR)
    footer_frame.pack(side="bottom", fill="x")

    disclaimer_text = """Disclaimer: The information provided by this application is generated automatically based on the input data and processed using Google's Gemini 1.5 model. While we strive to ensure accuracy, no guarantees are made regarding the correctness, completeness, or reliability of the data produced. This application and its creator do not assume responsibility for any errors, omissions, or inaccuracies in the results generated. The user should verify the information independently before making any decisions based on the provided output. In addition, as this application works using a Google AI API-KEY, the developer reserves the right to remove the KEY if the costs of the API change or if the developer deems it appropriate. By using this application, you acknowledge and agree to these terms."""

    footer_label = tk.Label(
        footer_frame,
        text=disclaimer_text,
        font=("Helvetica", 10),
        bg=BG_COLOR,
        fg="#ffffff",  # White text for contrast
        wraplength=root.winfo_screenwidth() - 40,  # Wrap text for readability
        justify="center",
        padx=10,
        pady=5
    )
    footer_label.pack()

    root.mainloop()


# ---Main method to call the user interface---
if __name__ == "__main__":
    # Ensure required folders exist
    utilities.ensure_folders_exist([
        "uploads_adjudication",
        "uploads_start",
        "results_adjudication",
        "results_start"
    ])

    # Verify internet connection
    is_connected = utilities.verify_internet_connection()

    # Launch the GUI
    main_gui()
