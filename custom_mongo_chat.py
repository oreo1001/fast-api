import json
import logging
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)
from pymongo import MongoClient, errors

logger = logging.getLogger(__name__)

DEFAULT_DBNAME = "chat_history"
DEFAULT_COLLECTION_NAME = "message_store"


class CustomMongoDBChatHistory(BaseChatMessageHistory):
    """Chat message history that stores history in MongoDB.

    Args:
        connection_string: connection string to connect to MongoDB
        session_id: arbitrary key that is used to store the messages
            of a single chat session.
        database_name: name of the database to use
        collection_name: name of the collection to use
    """

    def __init__(
        self,
        connection_string: str,
        session_id: str,
        database_name: str = DEFAULT_DBNAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.connection_string = connection_string
        self.session_id = session_id
        self.database_name = database_name
        self.collection_name = collection_name

        try:
            self.client: MongoClient = MongoClient(connection_string)
        except errors.ConnectionFailure as error:
            logger.error(error)

        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.collection.create_index("SessionId")

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from MongoDB"""
        try:
            cursor = self.collection.find({"SessionId": self.session_id})
        except errors.OperationFailure as error:
            logger.error(error)
            return []

        items = []
        if cursor:
            for document in cursor:
                history = document.get("History")
                if history:
                    # JSON을 파싱할 때 ensure_ascii=False로 저장된 한글을 그대로 사용
                    history_dict = json.loads(history)
                    items.append(history_dict)

        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in MongoDB"""
        try:
            message_dict = message_to_dict(message)
            self.collection.insert_one(
                {
                    "SessionId": self.session_id,
                    "History": json.dumps(message_dict, ensure_ascii=False),  # ensure_ascii=False 옵션 추가
                }
            )
        except errors.WriteError as err:
            logger.error(err)

    def clear(self) -> None:
        """Clear session memory from MongoDB"""
        try:
            self.collection.delete_many({"SessionId": self.session_id})
        except errors.WriteError as err:
            logger.error(err)