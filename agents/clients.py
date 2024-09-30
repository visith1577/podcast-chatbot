import groq 
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

groq_client = groq.Groq(
    api_key=os.getenv('GROQ_API_KEY'),
)

cohere_client = cohere.ClientV2(os.getenv("COHERE_API_KEY"))