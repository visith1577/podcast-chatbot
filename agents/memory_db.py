from zep_cloud.client import Zep
from zep_cloud import Message
from dotenv import load_dotenv
from agents.clients import groq_client
import ell
import uuid
import os

load_dotenv()

SESSION_ID = uuid.uuid4().hex

zep_client = Zep(
    api_key=os.environ.get('ZEP_API_KEY'),
)


def create_user(user_id: str, email: str, firstname: str, lastname: str):
    new_user = zep_client.user.add(
        user_id=user_id,
        email=email,
        first_name=firstname,
        last_name=lastname,
        metadata={},
    )
    return new_user

def create_session(user_id: str, session_id: str):
    new_session =  zep_client.memory.add_session(
        user_id=user_id,
        session_id=session_id,
        metadata={},
    )
    return new_session

def delete_session(session_id: str):
    zep_client.memory.delete(session_id)   

def delete_user(user_id: str):
    zep_client.user.delete(user_id)

def add_memory(session_id, user_content: str, assistent_content: str):
    memory = zep_client.memory.add(
        session_id=session_id,
        messages=[
            Message(
                content=user_content,
                role="user",
                role_type="user",
            ),
            Message(
                content=assistent_content,
                role="assistant",
                role_type="assistant",
            )
        ],
        #summary_instruction="Summarize the conversation highlighting the key points and the query discussed provide URLS, timestamps and youtube urls",
    )
    return memory

def get_memory(session_id):
    memory = zep_client.memory.get(session_id)
    return memory


@ell.simple(model="llama-3.1-8b-instant", client=groq_client)
def summarize_conversation(conversation: str) -> str:
    return [
        ell.system("""
        You are an expert transcriber. you will summarise a text containing a reply from a podcast host. Your summary must contain  what was spoken, who spoke about it and the timestamp and url in the format [timestamp](url).\n
        provide only the summary and nothing else.
        """),
        ell.user(conversation)
    ]