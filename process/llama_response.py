from ollama import chat, ChatResponse
    

def ollama_response_without_context(query: str):

    prompt = f"""
    Answer the following question with either "yes" or "no". Do not provide any explanations. Use lowercase.

    Question: {query}
    """

    response: ChatResponse = chat(model='llama3.2:3b', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    return response.message.content


def ollama_response_with_context(query: str, context: str):

    prompt = f"""
    The following is the relevant context extracted from a knowledge base:

    {context}

    Based on this context, answer the following question with either "yes" or "no". Do not provide any explanations. Use lowercase.

    {query}
    """

    response: ChatResponse = chat(model='llama3.2:3b', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    return response.message.content


def ollama_response_recommendations_without_context(query: str):
    prompt = f"""
    {query}

    Provide medical recommendations and a list of preventive measures.
    """

    response: ChatResponse = chat(model='llama3.2:3b', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    return response.message.content

def output_prettify(raw_output: str) -> str:
    # Replace `####` with a more structured markdown heading
    formatted_output = raw_output.replace("####", "###")

    # Add line breaks after bullet points and between sections
    formatted_output = formatted_output.replace("\n-", "\n\n-")
    formatted_output = formatted_output.replace("**Recommendations:**", "\n\n#### Recommendations:")
    formatted_output = formatted_output.replace("**Preventive Measures:**", "\n\n#### Preventive Measures:")

    # Ensure consistent spacing between sections
    formatted_output = "\n\n".join([section.strip() for section in formatted_output.split("\n\n")])

    # Return the pretty formatted output
    return formatted_output



def ollama_response_recommendations_with_context(query: str, context: str):
    prompt = f"""
    {query}

    The following is the relevant context extracted from a knowledge base:

    {context}

    Based on this context, provide medical recommendations and a list of preventive measures.
    """

    response: ChatResponse = chat(model='llama3.2:3b', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    
    content = response.message.content
    
    formatted_content = output_prettify(content)

    return formatted_content
