from dotenv import load_dotenv
from groq import Groq


load_dotenv("../.env")  # one level up
client = Groq()

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",        # cheap, fast, great for learning
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what a large language model is in 2 sentences."}
    ]
)

print(response.choices[0].message.content)
print(f"\nTokens used: {response.usage.total_tokens}")