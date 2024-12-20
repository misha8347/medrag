from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from typing import List

class SentenceTransformerSummarizer:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # For semantic similarity
        self.summarizer = pipeline("summarization")

    def process(self, query: str, context_snippets: List[str]):
        # query = "What is Natural Language Processing?"
        # context_snippets = [
        #     "Natural Language Processing (NLP) is a field of Artificial Intelligence that focuses on human-computer interactions.",
        #     "It enables machines to understand and generate human language effectively.",
        #     "NLP involves computational linguistics and deep learning.",
        #     "Artificial Intelligence aims to create intelligent systems, of which NLP is a part.",
        #     "Machine learning models are used for text analysis, understanding, and generation."
        # ]
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        context_embeddings = self.embedding_model.encode(context_snippets, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, context_embeddings).squeeze().tolist()

        ranked_snippets = sorted(
            zip(similarities, context_snippets), reverse=True, key=lambda x: x[0]
        )

        if len(ranked_snippets) < 10:
            num_snippets = len(ranked_snippets)
        else:
            num_snippets = 0.5 * len(ranked_snippets) 
        top_snippets = [snippet for _, snippet in ranked_snippets[:num_snippets]]
        top_context = " ".join(top_snippets)
        summary = self.summarizer(top_context, max_length=len(top_context), min_length=int(len(top_context) * 0.4), do_sample=False)

        print("Query:", query)
        print("Top Relevant Context:\n", top_context)
        print("Summary:\n", summary[0]['summary_text'])

        return query, top_context, summary[0]['summary_text']