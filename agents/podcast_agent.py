from agents.rag_functions import get_documents, graded_documents, llm_generation, halucinations_score, answer_grade, requery
from typing import List

RETRIEVAL_RETRIES = ANSWER_RETRIES = HALLUCINATION_RETRIES = 3


def rag_response(query: str, intent: str, output_sentiment: str, documents: List[str], history: str = None):
    """Generate an answer using the llm model.

    Args:
        query: The query text.
        intent: The user intent.
        output_sentiment: The output sentiment.
        documents: The list of documents.

    Returns:
        The generated answer from llm.
    """
    response, document = llm_generation(query=query, intent=intent, output_sentiment=output_sentiment, documents=documents, history=history)

    score = halucinations_score(documents=document, answer=response)

    if str(score).lower() == "no" and HALLUCINATION_RETRIES > 0:
        new_query = requery(query=query)
        return rag_response(query=new_query, intent=intent, output_sentiment=output_sentiment, documents=documents, history=history)
    return response


def rag_agent(query: str, user_intent: str, output_emotion: str, history: str = None):
    """RAG model for response generation.

    Args:
        query: The query text.
        user_intent: The user intent.
        output_emotion: The output sentiment.

    Returns:
        The generated response from the RAG model.
    """
    documents = get_documents(query=query, limit=5)

    graded_documents_list = graded_documents(query=query, documents=documents)

    # check if the documents are not relevant AND the retries are not exhausted
    if len(graded_documents_list) == 0 and RETRIEVAL_RETRIES > 0:
        requery = requery(query=query)
        return rag_agent(query=requery, user_intent=user_intent, output_emotion=output_emotion, history=history)

    response = rag_response(query=query, intent=user_intent, output_sentiment=output_emotion, documents=graded_documents_list, history=history)
    grade = answer_grade(answer=response, question=query)
    
    # check if the answer is not relevant AND the retries are not exhausted
    if str(grade).lower() == "no" and ANSWER_RETRIES > 0:
        new_query = requery(query=query)
        return rag_agent(query=new_query, user_intent=user_intent, output_emotion=output_emotion, history=history)

    return response

def execute_rag_response(query: str, user_intent: str, output_emotion: str, history: str = None):
    """Execute the RAG response.

    Args:
        query: The query text.
        user_intent: The user intent.
        output_emotion: The output sentiment.
    
    Returns:
        The response from the RAG model.
    """
    global RETRIEVAL_RETRIES, ANSWER_RETRIES, HALLUCINATION_RETRIES

    # init retries
    RETRIEVAL_RETRIES = ANSWER_RETRIES = HALLUCINATION_RETRIES = 3

    try:
        response = rag_agent(query=query, user_intent=user_intent, output_emotion=output_emotion, history=history)
    except Exception as e:
        response = f"An error occurred: {e}"
    finally:
        # reset retries
        RETRIEVAL_RETRIES = ANSWER_RETRIES = HALLUCINATION_RETRIES = 3
    
    return response
