import csv
import pandas as pd
from process import generate_response_without_context, ollama_response_without_context
from tqdm import tqdm

class PubmedQAEvaluation:
    def __init__(self, medrag_processor):
        self.medrag_processor = medrag_processor

    def evaluate(self, path_to_csv: str, is_with_context: bool = False, model_name: str = "llama"):
        assert model_name in ['llama', 'openai']
        items = pd.read_csv(path_to_csv).to_dict('records')
        records = []
        for item in tqdm(items):
            if not is_with_context:
                if model_name == 'openai':
                    response = generate_response_without_context(item['QUESTION'])
                else:
                    response = ollama_response_without_context(item['QUESTION'])
            else:
                response = self.medrag_processor.process(item['QUESTION'], model_name)
            record = {
                'QUESTION': item['QUESTION'],
                'label': item['final_decision'],
                'pred': response
            }
            records.append(record)
        
        return records