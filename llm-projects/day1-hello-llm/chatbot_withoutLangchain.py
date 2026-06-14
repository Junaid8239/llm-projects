from dotenv import load_dotenv
from groq import Groq

load_dotenv("../.env")
client = Groq()

conversation_history = []

system_message = {
    "role": "system",
    "content": "You are a concise, helpful AI assistant. Keep answers under 3 sentences unless asked for more."
}

print("Chatbot ready. Type 'quit' to exit, 'history' to see memory.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "quit":
        break
    if user_input.lower() == "history":
        print("\n--- Memory ---")
        for msg in conversation_history:
            role = "You" if msg["role"] == "user" else "Bot"
            print(f"{role}: {msg['content']}")
        print("--- End ---\n")
        continue
    if not user_input:
        continue

    conversation_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[system_message] + conversation_history
    )

    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})

    print(f"Bot: {reply}")
    # print(f"[memory object: {conversation_history}]\n")
    # total_chars = sum(len(m["content"]) for m in conversation_history)
    # print(f"[~{total_chars} chars in context so far]\n")
    print(f"[turns in memory: {len(conversation_history)//2}]\n")