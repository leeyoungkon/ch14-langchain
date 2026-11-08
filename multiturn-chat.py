from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Connect to Ollama
llm = ChatOllama(
    model="gemma3:12b",
    base_url="http://localhost:11435"
)

# 2. Configure the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that understands the context of previous conversations. Respond in English."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# 3. Create the chain
chain = prompt | llm

# 4. Maintain conversation history for each session
store = {}

def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 5. Attach conversation history to the chain
chatbot = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history"
)

# 6. Run a multi-turn conversation
config = {
    "configurable": {"session_id": "user001"}
}

while True:
    question = input("User: ")

    if question.lower() in ("exit", "quit"):
        break

    response = chatbot.invoke(
        {"question": question},
        config=config
    )

    print("AI:", response.content)