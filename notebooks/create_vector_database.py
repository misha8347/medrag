from datasets import load_dataset
import os
import sys

storage_path = os.path.abspath("../storages")
sys.path.append(storage_path)

from vector_db import VectorDB

def main():
    vector_database = VectorDB()
    ds = load_dataset("MedRAG/pubmed")
    print('dataset installed successfully!!!!!!')

    vector_database.create_knowledge_base(ds)

if __name__ == "__main__":
    main()