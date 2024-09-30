from agents.rag_db import get_results
from typing import Optional, List
from agents.clients import cohere_client
from agents.rag_agents import grade_answer, grade_document, check_halucinations, llm_answer, requery


def get_documents(query: str, limit: Optional[int]):
    """Get the response from the RAG model.

    Args:
        query: The query text.
        limit: The number of results to return.

    Returns:
        The documents retrieved from vector database.
    """
    results = get_results(query, collection_name="podcasts", limit=limit)
    # reranker
    # reranker = cohere_client.rerank(model="rerank-english-v3.0", query=query, documents=results, return_documents=False, top_n=5)
    # indices = [reranker[i].index for i in range(len(reranker))]
    # documents = [results[i] for i in indices]
    return results


def graded_documents(query: str, documents: List[str]):
    """Grade the documents based on the query.

    Args:
        query: The query text.

    Returns:
        The graded documents.
    """
    final_document_list = []
    for document in documents:
        grade = grade_document(query=query, document=document)
        if grade.content[-1].parsed.binary_score == "yes":
            final_document_list.append(document)
        else:
            continue

    return final_document_list


def llm_generation(query: str, intent: str, output_sentiment: str, documents: List[str], history: str):
    """Generate an answer using the LLM model.

    Args:
        query: The query text.
        intent: The user intent.
        output_sentiment: The output sentiment.
        documents: The list of documents.

    Returns:
        The generated answer from llm.
    """
    result = llm_answer(query=query, user_intent=intent, output_emotion=output_sentiment, documents=documents, history=history)
    return result, documents


def halucinations_score(documents: List[str], answer: str):
    """Check if the answer is hallucinated.

    Args:
        documents: The list of documents.
        answer: The answer text.

    Returns:
        The result of hallucination check.
    """
    result = check_halucinations(document=documents, answer=answer)
    return result.content[-1].parsed.binary_score


def answer_grade(answer: str, question: str):
    """Grade the answer based on the question. check if answer resolves the question.

    Args:
        answer: The answer text.
        question: The question text.

    Returns:
        The graded answer.
    """
    grade = grade_answer(answer=answer, question=question)
    return grade.content[-1].parsed.binary_score


def requery(query: str):
    """Requery the chatbot with the new query.

    Args:
        query: The query text.
        documents: The list of documents.

    Returns:
        The response from the chatbot.
    """
    new_query = requery(query)
    return new_query