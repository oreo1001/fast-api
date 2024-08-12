from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOllama(model="gemma2:9b-instruct-fp16")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and kind cafe order Assistant. You must answer in Korean.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm | StrOutputParser()

