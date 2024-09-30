import os
import ell 
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agents.memory_db import zep_client, SESSION_ID
from typing import List
from ell import Message

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

ell.init(store="./logdir")


class ChatbotEntry(BaseModel):
    answer: str = Field(..., description="answer should contain, the answer to user query, any URL links, timestamps and the speaker name.if answer is available in history")
    use_rag: bool = Field(..., description="Whether to use RAG model for response generation")
    user_intent: str = Field(..., description="User intent for the query")
    output_emotion: str = Field(..., description="Emotion the generated response should convey")

# long term memory through tool calling

# @ell.tool()
# def search_chatbot_history(query: str = Field(description="query to search the user long term history")) -> List[str]:
#     """search through chatbot history for previouse sessions.
#     Args:
#         query: The query text.
#     Returns:
#         The search results from the chatbot history.
#     """
#     data = zep_client.memory.search_sessions(
#         session_id=SESSION_ID,
#         text=query,
#         limit=3,
#         search_scope="messages",
#         search_type="mmr",
#         mmr_lambda=0.8,
#         user_id="user_1"
#     )
#     history = []
#     for search_results in data.results:
#         res = search_results.message.dict()
#         history.append(f"content: {res['content']} | role: {res['role']}")
    
#     return history 

@ell.complex(model="gpt-4o-2024-08-06", response_format=ChatbotEntry)
def chatbot_entry(query: str, history: List[Message], facts: str) -> ChatbotEntry:
    "Entry point to the chatbot agent"
    return [
        ell.system("""
        You are a helpful assistant who is a professional podcast host. 
        Your task is to provide a conversational, engaging, and context-aware answer to the query provided, while reflecting the tone and sentiment of the user’s input.\n
        Additionally, you will integrate disfluencies, informal language, and overlapping speech from the conversation when necessary, to maintain a natural and coherent podcast-style flow.
        follow the instructions within the <INS> tags.
        <INS>  
        - You are provided with a user query, a history of previous queries and responses, and a summary of the interaction history so far.  
        - Analyze the user query and history to understand user intent and sentiment, generating a response in a conversational, podcast-like tone. 
        - Use the previous responses and the summarised history to provide contextually relevant responses, maintaining the conversational flow. 

        Toxic Speech Handling: \n 
        - If you detect toxic or gibberish speech, acknowledge the sentiment and generate a meaningful response that reflects empathy or understanding. Respond in a way that maintains a conversational, respectful tone, just as a podcast host would manage heated or difficult conversations. For example, you might say: "I can sense there's frustration here, but let's keep this respectful and productive."  
        - In such cases, use_rag=False, user_intent=Toxic, output_emotion=None.  

        Using History:\n  
        - If the answer to the user query is available in the previous responses, generate a response based on the history, including the speaker name, timestamp, and YouTube link in the following format: [(02:16:41)](https://youtube.com/watch?v=tYrdMjVXyNg&t=8201).  
        - use_rag=False for these cases where a complete and concise answer can be derived from the history.  
        - if the answer is available but user explicitly asks to use find more information / explain or if you feel the answer in history is vague or irrelevant then use_rag=True.
        - Maintain the conversational flow of a podcast host, keeping the tone natural and engaging.  

        New Queries:  
        - If the answer is not available in the history, output 'no answer' and enable RAG retrieval with use_rag=True.  

        The output should contain:  
        - The generated response with speaker name, timestamp, and URL [timestamped YouTube link] (if possible). only provide speaker name, timestamp or URL if available in history else if you do not know, express it,  
        - The use_rag condition (True/False),  
        - The user intent (e.g., Information Request, Toxic),  
        - The output emotion (e.g., Empathy, Neutral, None).  
        </INS>
        """)
    ] + history + [ell.user(f"Output answer should contain, the answer to user query, any URL links, timestamps and the speaker name.\n\nQuery: {query} \n\n Summarised History: \n {facts}\n")]
