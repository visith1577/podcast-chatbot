from qdrant_client import QdrantClient
from qdrant_client.http import models
import openai
import groq
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os

load_dotenv()

openai_client = openai.Client()
openai.api_key = os.getenv("OPENAI_API_KEY")
VECTOR_SIZE = 1536

KEYWORD_PROMPT="""
Your task is to analyse the query and identify the entities in the query.
The output must contain only the entities separated by comma and no other details. 
Do not share anything other than what you are asked to.
You must strictly follow the instruction.
only provide the keywords found and nothing else.
"""


client = QdrantClient(
    url="https://3973cdf9-4ba6-40b1-ae92-b2f952f82fb9.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.getenv("QDRANT_CLOUD_API_KEY"),
)

 

def get_embedding(text: str) -> List[float]:
    """Get OpenAI embedding for the given text."""

    response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

groq_client = groq.Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

def get_entities(text: str) -> List[str]:
    """Get entities from the given text using GROQ."""
    response = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": KEYWORD_PROMPT}, {"role": "user", "content": text}],
        model="llama3-8b-8192",
    )
    return response.choices[0].message.content.split(", ")

def hybrid_search(
    collection_name: str,
    query: str,
    limit: int = 5,
    subtopic: Optional[str] = None,
    speakers: Optional[List[str]] = None,
    title: Optional[str] = None,
    full_text_search: bool = True
) -> List[Dict]: 
    """Search for similar documents in the collection using a hybrid search approach.
    
    Args:
        collection_name: The name of the collection to search in.
        query: The query text.
        limit: The number of results to return.
        subtopic: The subtopic of the document.
        speakers: The speakers of the document.
        title: The title of the document.

    Returns:
        A list of dictionaries containing the search results.
    """

    # Get the embeddings for the query text.
    query_embedding = get_embedding(query)

    must_conditions = []
    should_conditions = []

    final_result = []

    # Metadata filtering
    if subtopic:
        must_conditions.append(models.FieldCondition(key="subtopic", match=models.MatchValue(value=subtopic)))
    if speakers:
        must_conditions.append(models.FieldCondition(key="metadata.speakers", match=models.MatchAny(any=speakers)))
    if title:
        must_conditions.append(models.FieldCondition(key="metadata.title", match=models.MatchValue(value=title)))

    # Full-text search condition
    if full_text_search == True:
        entities = get_entities(query)
        for word in entities:
            should_conditions.append(models.FieldCondition(key="content", match=models.MatchText(text=word)))

    # search with and without full-text search

    if full_text_search == True:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=models.Filter(
                must=must_conditions,
                should=should_conditions
            ),
            limit=limit,
            with_payload=True,
            score_threshold=0.0
        )
        final_result = search_result

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        query_filter=models.Filter(
            must=must_conditions
        ),
        limit=limit,
        with_payload=True,
        score_threshold=0.0
    )
    final_result += search_result
    
    retrieved_docs = [
        {
            "id": hit.id,
            "subtopic": hit.payload.get("subtopic"),
            "speakers": hit.payload.get("speakers"),
            "content": hit.payload.get("content"),
            "title": hit.payload.get("title"),
            "url": hit.payload.get("url"),
            "timestamp": hit.payload.get("timestamp"),
            "score": hit.score
        }
        for hit in final_result
    ]

    # remove duplicates and sort by score
    seen = set()
    unique_docs = []
    for doc in retrieved_docs:
        if doc["id"] not in seen:
            seen.add(doc["id"])
            unique_docs.append(doc)
    unique_docs = sorted(unique_docs, key=lambda x: x["score"], reverse=True)
    return unique_docs

def markdown_template(data) -> str:
    RESULT_TEMPLATE=f"""
    **Document**:
    - **Title**: {data['title']}\n
    - **Subtopic**: {data['subtopic']}\n
    - **Speakers**: {', '.join(data['speakers'])}\n
    - **Timestamp**: {data['timestamp']}\n
    - **URL**: [{data['url']}]\n

    **Content**:\n
    {data['content']}
    """
    return RESULT_TEMPLATE

def get_results(query: str, collection_name: str, limit: int = 5) -> List[str]:
    """Get search results for the given query."""
    results = hybrid_search(collection_name=collection_name, query=query, limit=limit)
    return [markdown_template(data) for data in results]