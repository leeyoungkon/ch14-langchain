from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Ollama 연결
llm = ChatOllama(
    model="gemma3:12b",
    base_url="http://localhost:11434"
)

# 2. 프롬프트 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 이전 대화 맥락을 이해하는 AI 비서입니다."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# 3. Chain 생성
chain = prompt | llm

# 4. 세션별 대화 이력
store = {}

def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 5. 대화 이력 연결
chatbot = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history"
)

# 6. 멀티턴 대화
config = {
    "configurable": {"session_id": "user001"}
}

while True:
    question = input("사용자: ")

    if question.lower() in ("exit", "quit"):
        break

    response = chatbot.invoke(
        {"question": question},
        config=config
    )

    print("AI:", response.content)