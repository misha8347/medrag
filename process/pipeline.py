import os
import sys

storage_path = os.path.abspath("../storages")
summarization_path = os.path.abspath("../summarization")
sys.path.append(storage_path)
sys.path.append(summarization_path)

from vector_db import VectorDB, convert_documents_to_text
from text_rank import text_rank_summarization
from llm_response import generate_response_with_context
from llama_response import ollama_response_with_context

class MedRAGPipeline:
    def __init__(self):
        self.vector_database = VectorDB()
        self.vector_database.load_db()

    def process(self, query: str, model_name: str):
        assert model_name in ['llama', 'openai']

        search_results = self.vector_database.search(query, k=15)
        context = convert_documents_to_text(search_results)
        # summarized_context = text_rank_summarization(context)
        
        if model_name == 'openai':
            return generate_response_with_context(query, context)
        return ollama_response_with_context(query, context)