import requests

def query_ollama(prompt, model="mistral:7b-instruct-q4_0"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # Set to True for streaming responses
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    return response.json()['response']

# Example usage
response = query_ollama("Explain quantum computing in simple terms.")
print(response)
