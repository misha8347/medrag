from openai import OpenAI
import os
from dotenv import load_dotenv


def generate_response_without_context(query):
    # load_dotenv('/etc/.asr')
    # openai_api_key = os.getenv('OPENAI_API_KEY')
    # client = OpenAI(api_key=openai_api_key)

    prompt = f"""
    Answer the following question with either "yes" or "no". Do not provide any explanations. Use lowercase.

    Question: {query}
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the provided context, and if it's insufficient, rely on your general knowledge."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=5,
        temperature=0
    )

    return completion.choices[0].message.content.strip()


def generate_response_with_context(query, context):
    # load_dotenv('/etc/.asr')
    # openai_api_key = os.getenv('OPENAI_API_KEY')
    # client = OpenAI(api_key=openai_api_key)

    prompt = f"""
    The following is the relevant context extracted from a knowledge base:

    {context}

    Based on this context, answer the following question with either "yes" or "no". Do not provide any explanations. Use lowercase.

    {query}
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the provided context, and if it's insufficient, rely on your general knowledge."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=5,
        temperature=0
    )

    return completion.choices[0].message.content
