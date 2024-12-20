from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import pandas as pd
from loguru import logger
import os
from langchain_core.documents import Document
import torch
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Tuple
from datasets import DatasetDict

#https://www.anthropic.com/news/contextual-retrieval
#https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings

def convert_documents_to_text(results: List[Tuple[Document, float]]) -> str:
    context = "\n\n".join(doc.page_content for doc, score in results)
    return context

class VectorDB:
    def __init__(self, 
                 db_path: str = '/s3/misha/data_dir/MedRAG/db_faiss', 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)

        self.index_path = os.path.join(self.db_path, "faiss_index.idx")
        self.metadata_path = os.path.join(self.db_path, "metadata.pkl")

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': self.device})


    def create_knowledge_base(self, hf_dataset: DatasetDict):
        df = hf_dataset['train']
        logger.info('Encoding texts into embeddings...')

        texts = df["contents"]
        
        texts_with_progress = tqdm(texts, desc="Embedding texts")
        embeddings_list = [self.embeddings.embed_query(text) for text in texts_with_progress]
        text_embedding_pairs = list(zip(texts, embeddings_list))
        self.vector_store = FAISS.from_embeddings(text_embeddings=text_embedding_pairs, embedding=self.embeddings)

        logger.info('Saving vector store...')
        self.vector_store.save_local(self.db_path)
        logger.info('Vector store saved successfully!')

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found. Use upload_data to create a new database.")

        self.vector_store = FAISS.load_local(self.db_path, 
                                             embeddings=self.embeddings, 
                                             allow_dangerous_deserialization=True)
        
    def search(self, query: str, k: int):
        if not self.vector_store:
            raise ValueError("Vector database is not created. Use create_knowledge_base() to create a new database.")
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)

        return results
        

    