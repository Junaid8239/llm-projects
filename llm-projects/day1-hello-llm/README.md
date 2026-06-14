# Day 1 — Hello LLM + Chatbot with Memory

First project in my GenAI Engineering journey. A conversational chatbot with persistent memory built using LangChain + Groq.

## What it does
- hello_llm.py — first raw LLM API call, prints response + token count
- chatbot.py — simple chatbot with manual conversation history
- chatbot_withLangchain.py — same chatbot using LangChain RunnableWithMessageHistory

## Key concepts learned
- LLMs are stateless — memory works by sending full history every turn
- LangChain RunnableWithMessageHistory automates history injection
- ChatPromptTemplate structures system + history + user message
- MessagesPlaceholder injects conversation history into the prompt

## Stack
Python · LangChain · LangChain-Groq · Groq API · Llama 3.1 8B

## Run it
```bash
source ../venv/bin/activate
python hello_llm.py
python chatbot_withLangchain.py
```

## Interview answer
"I built a conversational chatbot using LangChain's RunnableWithMessageHistory. The key insight is that LLMs are stateless — memory works by appending the full conversation history to every new prompt. LangChain automates this with MessagesPlaceholder. The tradeoff is token cost — longer conversations mean more tokens sent every turn."