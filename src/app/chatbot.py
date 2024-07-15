import json
import os
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.chat import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from constant.chatbot_text import ASSESS_PROMPT_TEMPLATE


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


class UserAnalyzer:
    def __init__(self, llm) -> None:
        self.llm = llm
        system_template = ASSESS_PROMPT_TEMPLATE
        human_template = "{input}"
        self.expected_output = [
            "aggressive",
            "moderate-aggressive",
            "moderate",
            "moderate-conservative",
            "conservative",
        ]

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_template),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(human_template),
            ]
        )

        print(chat_prompt)
        # 3. Define the conversation chain
        chain = chat_prompt | self.llm
        self.conversation = RunnableWithMessageHistory(
            chain,
            # Uses the get_by_session_id function defined in the example
            # above.
            get_by_session_id,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        self.user_input = None
        self.ask_user = True
        self.bot_response = {"response_status": "question", "response": None}
        self.session_id = "0"

    def predict(self, prompt: str):
        response = self.conversation.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": self.session_id}},
        )

        try:
            response = json.loads(response.content)
        except json.JSONDecodeError:
            # Placeholder
            print(
                f"WARNING: Invalid decoding to dict from output text: {response.content}"
            )
            response = {"response_status": "question", "response": response.content}
        self.bot_response = response
        self.bot_response["response_status"] = self.bot_response[
            "response_status"
        ].lower()

    def get_history(self):
        return self.conversation.get_session_history(session_id=self.session_id)

    def __call__(self, user_input) -> Dict[str, Any]:
        if user_input is None:
            prompt = "Start the conversation with the first question."
            self.predict(prompt)
            return self.bot_response

        # Ask question based on previous bot answer
        if self.bot_response["response_status"] == "question":
            prompt = f"User: {user_input}.\n Ask next question, or provide result if you have enough information."
            self.predict(prompt)
        elif self.bot_response["response_status"] == "explanation":
            prompt = f"User: {user_input}.\n Response to what user asked. Follow by asking previous question again."
            self.predict(prompt)
        elif self.bot_response["response_status"] == "unrelated":
            prompt = f"User: {user_input}.\n "
        elif (
            self.bot_response["response_status"] == "result"
        ):  # This only reach if result content is not statisfied.
            prompt = f"Please rephrase result again. Give exactly one of the following answers:  Aggressive, Moderate-aggressive, Moderate, Moderate-conservative, Conservative. No extra response!! "
            self.predict(prompt)
        else:
            prompt = f"Please rephrase your previous result again and return response_status value with exactly one of followings: 'Question', 'Explanation', 'Result'"
            self.predict(prompt)

        return self.bot_response


class MofuChatBot:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(temperature=0.7)
        self.conversations = {
            "analyze-user": UserAnalyzer(self.llm),
            "confirm-equity": None,
        }
        self.mode = "analyze-user"
        self.current_conversation = self.conversations["analyze-user"]

    def reset(self):
        self.conversations = {
            "analyze-user": UserAnalyzer(self.llm),
            "confirm-equity": None,
        }
        self.mode = "analyze-user"
        self.current_conversation = self.conversations["analyze-user"]

    def set_status(self, mode: str):
        """
        Mode: ["analyze-user", "confirm-equity"]
        """
        if mode in self.conversations:
            self.current_conversation = self.conversations[mode]
            self.mode = mode
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected: {self.conversations}")

    def chat(self, user_input: str):
        output = self.current_conversation(user_input)
        print("Input of chat", (user_input))
        print("Output of chat", output)
        return output["response"]
