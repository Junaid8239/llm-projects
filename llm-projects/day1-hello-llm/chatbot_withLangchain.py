from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv("../.env")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a concise helpful assistant. Keep answers under 3 sentences unless asked for more."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

history = ChatMessageHistory()

chain = RunnableWithMessageHistory(
    prompt | llm,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history"
)

print("LangChain chatbot ready. Type 'quit' to exit, 'history' to see memory.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "quit":
        break
    if user_input.lower() == "history":
        print("\n--- Memory ---")
        for msg in history.messages:
            role = "You" if msg.type == "human" else "Bot"
            print(f"{role}: {msg.content}")
        print("--- End ---\n")
        continue
    if not user_input:
        continue

    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "default"}}
    )
    print(f"Bot: {response.content}")
    print(f"[turns in memory: {len(history.messages)//2}]\n")