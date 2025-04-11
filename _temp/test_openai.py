import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(r'G:\Shared\proyek\rag_exercise\rag_kuhp\.env')

# Get API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key not found in .env file")

client = OpenAI(
    api_key=api_key
)

response = client.responses.create(
    model='gpt-3.5-turbo',  # or "gpt-3.5-turbo"
    input='Apa isi pasal KUHP tentang pencurian?'
)

# Print the response
print(response.output_text)
