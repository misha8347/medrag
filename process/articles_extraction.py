from ollama import chat
from ollama import ChatResponse
import re
import requests
from Bio import Entrez
from typing import Dict, List
import sys
import os
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarization.text_rank_summarization import QueryBasedTextRankSummarizer


def extract_symptoms(query: str):
    prompt = f"""
    Identify and list all symptoms mentioned in the following text. 
    Include only the words or phrases directly describing physical, emotional, or psychological conditions, signs, or ailments experienced by a person. Ignore unrelated details. 
    Present the symptoms as a clear, concise list. Here's the text:

    {query}
    """

    response: ChatResponse = chat(model='llama3.2:3b', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    return response.message.content

def extract_medical_terms(query: str):

    prompt = f"""
    Identify and list all medical terms and diseases mentioned in the following text. 
    Include only specific terms that refer to medical conditions, imaging techniques, procedures, anatomical structures, medical interventions, radiotracers, or any other specialized terminology used in medicine. Ignore unrelated details or general phrases. 
    Present the medical terms as a clear, concise list. Here's the text:

    {query}
    """

    response: ChatResponse = chat(model='llama3.2:3b', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    return response.message.content


def extract_top_symptoms(text, top_n=10) -> Dict:
    pattern = r'\d+\.\s(.+?)(?=\n|$)'
    symptoms = re.findall(pattern, text)
    
    prioritized_symptoms = [symptom.strip() for symptom in symptoms if symptom]
    
    result = prioritized_symptoms[:top_n]
    
    if len(result) < top_n:
        return {
            "symptoms": result,
            "note": f"Only {len(result)} symptoms found. Less than {top_n} symptoms available."
        }
    return {"symptoms": result}

def extract_relevant_article_ids(response: Dict) -> Dict:
    id_list_by_query = {}
    for i in range(len(response['symptoms'])):
        for j in range(i+1, len(response['symptoms'])):
            query = ' '.join([response['symptoms'][i], response['symptoms'][j]])
            # print(f'{i}. {query}')
            handle = Entrez.esearch(db="pubmed", term=query, retmax=10)
            record = Entrez.read(handle)
            handle.close()

            ids = record["IdList"]

            if len(ids) != 0:
                id_list_by_query[query] = ids

    return id_list_by_query

def clean_context_snippets(context_snippets: List[str]) -> List[str]:
    cleaned_snippets = [
        snippet.strip() for snippet in context_snippets
        if isinstance(snippet, str) and snippet.strip()
    ]
    return cleaned_snippets

def extract_full_texts_by_id(id_list_by_query: Dict):
    json_files = []

    for query, id_list in id_list_by_query.items():
        count = 0
        for id in id_list:
            response = requests.get(f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{id}/unicode')
            if 'No record can be found for the input' in response.text:
                # print("No record found. Continuing...")
                pass
            else:
                data = response.json()
                json_files.append(data)
                break

    total_texts_length = 0
    articles_and_full_texts = {}
    for i in range(len(json_files)):
        full_text_pieces = []
        json_file = json_files[i]
        id = json_file[0]['documents'][0]['id']
        passages = json_file[0]['documents'][0]['passages']

        for passage in passages:
            total_texts_length += len(passage['text'])
            full_text_pieces.append(passage['text'])
        
        articles_and_full_texts[id] = full_text_pieces

    return articles_and_full_texts, total_texts_length

def rank_summarized_texts(query: str, articles_and_summarized_texts: Dict):
    summarized_texts = list(articles_and_summarized_texts.values())

    documents = summarized_texts + [query]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    query_vector = tfidf_matrix[-1]
    article_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(query_vector, article_vectors).flatten()

    # Step 4: Rank articles by similarity
    ranked_indices = similarities.argsort()[::-1]
    ranked_articles = [(summarized_texts[i], similarities[i]) for i in ranked_indices]

    filtered_articles = [(article, score) for article, score in ranked_articles if score >= 0.1]
    top_articles = filtered_articles[:10] if len(filtered_articles) > 10 else filtered_articles

    return top_articles


class SummarizedArticlesExtractor:
    def __init__(self):
        Entrez.email = "ogai.misha@gmail.com"
        self.text_rank_summarizer = QueryBasedTextRankSummarizer()

    def process(self, query: str, task: str = 'diagnostic'):
        assert task in ['qa', 'diagnostic']
        if task == 'qa':
            response = extract_medical_terms(query)
        else:
            response = extract_symptoms(query)
        logger.info(f'extracted keywords: {response}')
        
        top_symptopms = extract_top_symptoms(response)
        logger.info(f'extracted top symptoms: {top_symptopms}')
        
        id_list_by_query = extract_relevant_article_ids(top_symptopms)
        logger.info('extracted relevant article ids')

        articles_and_full_texts, total_texts_length = extract_full_texts_by_id(id_list_by_query)
        logger.info(f'extracted full texts with the length {total_texts_length}')

        total_summarized_texts_length = 0
        articles_and_summarized_texts = {}
        for id, full_text in articles_and_full_texts.items():
            summarized_text = self.text_rank_summarizer.process(query, full_text)
            total_summarized_texts_length += len(summarized_text)
            articles_and_summarized_texts[id] = summarized_text
        logger.info(f'summarized full texts with the length {total_summarized_texts_length}')

        if len(articles_and_summarized_texts) != 0:
            top_relevant_articles = rank_summarized_texts(query, articles_and_summarized_texts)
        else:
            top_relevant_articles = []

        ranked_articles_text = ''
        max_length = 50000
        for rank, (article_summary, score) in enumerate(top_relevant_articles, 1):
            print(f"Rank {rank}:")
            print(f"Relevance Score: {score:.4f}")
            
            if len(ranked_articles_text) + len(article_summary) + 1 > max_length:
                break
            ranked_articles_text += ' ' + article_summary

        return ranked_articles_text

def main():

    query = """A 44-year-old female with a history of asthma, essential hypertension, class 3 obesity, depression, and poor social and economic background was intermittently followed during the previous four years for persistent cutaneous candidiasis with intertrigo in the inframammary, inguinal, and lower abdominal regions (Figure ).\nShe had been treated with topical antifungal, oral fluconazole and oral itraconazole with no improvement, which was believed to be because of poor hygiene and questionable therapeutic compliance. A worsening in the skin rash with exudate, pruritus, and a change to a violaceous colour, with scaly papules and vesicles (Figures , ) led to the performance of a skin biopsy which revealed (Figure ) orthokeratotic hyperkeratosis in the epidermis with areas of parakeratosis and, in the papillary dermis, there was an infiltrate of cells with eosinophilic cytoplasm and reniform nuclei that showed positive CD1a and S100 proteins on the immunohistochemistry and negative CD163 (Figure ).\nThe patient denied constitutional, musculoskeletal, neurological, or urinary complaints. She underwent a complete blood count, complete metabolic panel, brain magnetic resonance imaging (MRI), thoracic-abdominal-pelvic computed tomography (CT), and bone scintigraphy. Brain MRI depicted mild chronic microvascular changes in the white matter, unchanged from a prior study. CT demonstrated a thickening of the renal pelvis (4 mm) in the right kidney with a slight urothelial dilation (Figure ). The rest of the exams did not reveal further organ involvement.\nAfter considering the skin histology, the extensive cutaneous involvement, and the infiltrative urothelial involvement, it was evident this was a multi-system process. A consultation with Hematology/Oncology, led to induction treatment with prednisolone and vinblastine-based chemotherapy. At six weeks of chemotherapy, there was a partial regression of the skin lesions (Figure ) and a resolution of the urothelium lesion in imaging exam (CT).\nThe disease was in continuous regression and considering the extension of affected skin tissue a second round of chemotherapy with prednisolone and vinblastine was administered for six weeks. There was a resolution of all the lesions following this second round, and the patient underwent maintenance therapy consisting of administrating mercaptopurine daily and prednisolone/vinblastine every three weeks during 12 months, staying in remission (Figure ).\nSix months after the end of maintenance therapy the patient had a recurrence of the disease and started second-line chemotherapy with clofarabine and cytosine arabinoside (ARA-C). The patient did not comply with the treatment and the disease progressed. As a result of skin ulceration, she developed skin and soft tissue infection that evolved into septic shock and did not survive despite intensive care support."""

    summarized_articles_extractor = SummarizedArticlesExtractor()
    summarized_text = summarized_articles_extractor.process(query)

    print(len(summarized_text))

if __name__ == "__main__":
    main()