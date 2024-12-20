from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer, util
from sumy.summarizers.text_rank import TextRankSummarizer
from nltk.tokenize import sent_tokenize
from typing import List
import torch
# import nltk
# nltk.download('punkt')

def text_rank_summarization(text: str):
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)

    summary_sentence_count = max(1, round(0.2 * total_sentences))

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count=summary_sentence_count)

    summarized_text = ''.join(sentence._text for sentence in summary)

    # print(summarized_text)
    return summarized_text

# text = "Transient ischemic attacks in the elderly: diagnosis and treatment. Transient ischemic attacks (TIAs) are the most reliable warning sign of impending stroke and are highly indicative of significant coronary artery disease. The history and physical examination may suggest the pathologic mechanism, an important clue to diagnosis and prognosis. Diagnostic testing is individualized but often includes ECG and cerebral contrast angiography. Exercise testing, echocardiography, ultrasound, CT, and/or MRI are sometimes indicated. The patient with recent TIAs may be hospitalized for acute management. Long-term treatment includes stroke risk factor modification, use of antiplatelet agents, and sometimes anticoagulant therapy. Selected older patients may be candidates for carotid endarterectomy.Medical management of acute cerebral ischemia. Ischemic stroke is a major cause of death and disability. Despite its high incidence, acute management remains controversial. Most current forms of therapy are designed to reduce complications of a recent stroke or prevent recurrences. Experimental data suggest that the optimal time for intervention is the hour immediately following brain ischemia.Clinical aspects of cerebral ischemia. Out of 1,120 cases with focal cerebrovascular lesions, 102 cases (9.1%) were classified as transient episodes under the heading of reversible ischemic attacks (RIA). RIA comprises classical transient ischemic attacks (TIA), clearing completely within 24 h, usually few minutes or hours, and strokes in which a full recovery takes place over an average of 3 weeks. The clinical definition of transient and/or reversible requires a complete negative neurological examination including the normalization of the EEG and brain scan as well as a normal computed tomography. Even multiple transient episodes quite indistinguishable from classical TIA can be brought about by cerebral tumors. TIA are important symptoms, not a disease in themselves. Hypertension and cardiac disease along with carotid stenosis and/or occlusion seem to constitute the main conditions responsible for an evolution from TIA to completed strokes.Associated systemic factors in cerebrovascular ischemia. Systemic disorders (eg, cardiac, hematologic) are commonly recognized as predisposing and sometimes actual precipitating events in cerebral ischemia. From available studies, the incidence of precipitation is not clear. To determine this, we undertook a comprehensive investigation of all patients with ischemic brain disease for a one-year period. Results reveal that brain ischemia is more commonly precipitated by systemic illness than usually supposed, particularly transient ischemic attacks of the vertebrobasilar circulation and completed infarcts in the carotid distribution. Cardiac disorders outnumber all other precipitating events. As they are more amenable to therapy than atherosclerosis, a diligent search for such precipitating events is warranted in patients with ischemic symptoms.[Comparative clinical characteristics of ischemic stroke with reversible and stable neurological deficit]. Fifty-four patients with minor and 54 with complete brain stroke were examined for the effect of the patients' age, nature of the underlying disease, its course, acuity of development, initial intensity, and successive dynamics of focal symptoms on the course of brain stroke. It has been discovered that the patients' sex or age, or the nature of the underlying disease do not produce any effect on the development of minor or complete brain stroke. The tendency towards decrease of the intensity of focal symptoms within the first 24 hours since the disease onset and early hospitalization of the patients was of paramount importance in cases with a complete regression of focal symptoms. The highest percentage of minor brain strokes was recorded if the patients were hospitalized within 12 hours since the disease manifestation."
# print(text)
# print(len(text))
# print(len(text_rank_summarization(text)))

class QueryBasedTextRankSummarizer:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model.to(self.device)

    def process(self, query: str, context_snippets: List[str]):
        context_snippets = [snippet for snippet in context_snippets if snippet.strip()]
        if not context_snippets:
            raise ValueError("context_snippets must be a non-empty list of valid sentences.")

        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(self.device)
        sentence_embeddings = self.embedding_model.encode(context_snippets, convert_to_tensor=True).to(self.device)

        similarities = util.cos_sim(query_embedding, sentence_embeddings).squeeze().tolist()

        ranked_sentences = sorted(
            zip(similarities, context_snippets), reverse=True, key=lambda x: x[0]
        )

        count_ranked_sentences = len(ranked_sentences)
        filtered_context = " ".join([sentence for _, sentence in ranked_sentences[:int(count_ranked_sentences / 2)]])

        summary_sentence_count = max(1, round(0.2 * count_ranked_sentences))
        parser = PlaintextParser.from_string(filtered_context, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, summary_sentence_count)

        summarized_text = ''.join(sentence._text for sentence in summary)
        return summarized_text

