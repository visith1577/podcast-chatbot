import os
import ell 
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agents.clients import groq_client
from typing import List

load_dotenv()

ell.init(store="./logdir")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



# This module contains the output format for the agents.

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
    


# This module contains the agents for the RAG model.

@ell.complex(model="gpt-4o-mini", response_format=GradeDocuments)
def grade_document(query: str, document: str) -> GradeDocuments:
    "Document grading agent, grades document as yes or no"
    return [
        ell.system("""You are a grader assessing relevance of a retrieved document to a user question. \n 
                It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
                """),
        ell.user(f"Query: {query} \n\n Document: {document}"),
    ]


@ell.complex(model="gpt-4o-mini", response_format=GradeHallucinations)
def check_halucinations(document: List[str], answer: str) -> GradeHallucinations:
    "hallucination grading agent, grades answer as yes or no"
    formatted_document = "\n".join(document)
    return [
        ell.system("""
            You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts else 'no'.
            The response can be 'yes' or 'no' and nothing else.
                """),
        ell.user(f"Set of facts: \n\n {formatted_document} \n\n LLM generation: {answer}"),
    ]


@ell.complex(model="gpt-4o-mini", response_format=GradeAnswer)
def grade_answer(answer: str, question: str) -> GradeAnswer:
    "Answer grading agent, grades answer as yes or no"
    return [
        ell.system("""
            You are a grader assessing whether an answer addresses / resolves a question \n 
            You do not need to be overly strict. The goal is to filter out if irrelevant answers created. \n
            as long as the answer is relevant to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question else 'no'.
                """),
        ell.user(f"Question: {question} \n\n Answer: {answer}"),
    ]


@ell.simple(model="gpt-4o-2024-08-06")
def llm_answer(query: str, user_intent: str, output_emotion: str, documents: List[str], history: str) -> str:
    """Generate an answer using the LLM model.

    Args:
        query: The query text.
        user_intent: The user intent.
        output_emotion: The output sentiment.
        documents: The list of documents.

    Returns:
        The generated answer from llm.
    """
    formatted_documents = "\n".join(documents)
    return [
        ell.system("""
        You are a state-of-the-art Q&A chatbot designed to respond in the persona of a podcast host. 
        Your task is to provide a conversational, engaging, and context-aware answer to the query provided, while reflecting the tone and sentiment of the user’s input.\n
        Additionally, you will integrate disfluencies, informal language, and overlapping speech from the conversation when necessary, to maintain a natural and coherent podcast-style flow.
        follow the instructions within the <INS> tags.
        <INS>
        Answer Development:\n
        - Read the provided query, user intent, output emotion, and retrieved documents.
        - Analyze the sentiment and tone of the query (whether it’s humorous, sarcastic, angry, or neutral).
        - Formulate an initial response based on the query and provided documents.
        - Refine your response by reflecting on the user’s intent and emotional state, ensuring it matches the appropriate tone (empathetic, humorous, casual, etc.).
        - Read the pointwise conversation history to understand the context and ensure your response aligns with the ongoing discussion.\n\n
        - Use a conversational, flowing style that feels natural, as if you're speaking on a podcast. Incorporate slight pauses, filler words, or casual transitions if they enhance the flow.
        - If necessary, handle overlapping speech or informal language to ensure a smooth response.\n\n
        
        Sentiment and Tone Adjustment:\n
        - If the user’s sentiment is negative (e.g., anger or frustration), respond with a calm and empathetic tone, de-escalating the situation.
        - If the user is being humorous or sarcastic, mirror that tone with a witty or lighthearted response.
        - If the query is neutral or professional, maintain a balanced and informative tone.\n\n
                   
        Source Attribution:
        - Your retrived data consist of transcript url given by **URL**. additionally you can identify timestamp followed by the youtube link to time stamp.
        - Always include the YouTube video link and timestamp related to the source of your answer. If multiple timestamps are relevant, cite the most accurate one.
        - The generated response with speaker name, timestamp, and URL [timestamped YouTube link] provide link in the following format: [(time)](youtube url).
                   
        **Things to Remember**
        - You are a podcast host—keep the conversation engaging, natural, and suited to the user’s emotional state.
        - Handle conversational disfluencies and informal speech as part of your persona.
        - Always provide correct source attribution with YouTube links and timestamps.
        </INS>
        """),
        ell.user(f"Documents:\n {formatted_documents} \n\nConversation history: {history} \n\nQuery: {query} \n User Intent: {user_intent} \n answer with output emotion: {output_emotion}"),
    ]


@ell.simple(model="llama3-70b-8192", client=groq_client)
def requery(query: str) -> str:
    """Requery the chatbot with the new query.

    Args:
        query: The query text.
        documents: The list of documents.

    Returns:
        The response from the chatbot.
    """
    return [
        ell.system("""
        You are given a user query. You must requery it and provide with a new query so that relevant documents can be retrieved.
        Your output should strictly contain only the new query.
        """),
        ell.user(f"previose query: {query}"),
    ]