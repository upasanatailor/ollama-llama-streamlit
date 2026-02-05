import requests
import json


response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "gemma3",
        "prompt": "write a short story about a robot learning to love",
        "stream": False
    }
)

print(response.json()["response"])

