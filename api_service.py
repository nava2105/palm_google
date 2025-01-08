# api_service.py

import os
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv


# noinspection PyMethodMayBeStatic
class APIService:
    def __init__(self):
        """
        Initialize the APIService and load environment variables.
        """
        load_dotenv()
        self.configure_api()

    def configure_api(self):
        """
        Configures the API key for Google Generative AI.
        """
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if google_api_key:
            genai.configure(api_key=google_api_key)

    def generate_text_embedding(self, text, title="Text Chunk"):
        """
        Generates embeddings for the provided text using Google's embedding model.
        """
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
                title=title
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None

    def generate_response_from_embeddings(self, question, embeddings_with_chunks, model_name='gemini-1.5-flash'):
        """
        Generates a response using the question and relevant embeddings.
        """
        try:
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
        except Exception as e:
            print(f"Error generating response: {e}")
            return None