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

from constant.chatbot_text import *


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


class RiskAssessBot:
    def __init__(self, llm, session_id) -> None:
        self.llm = llm
        self.session_id = session_id
        self.expected_output = [
            "aggressive",
            "moderate-aggressive",
            "moderate",
            "moderate-conservative",
            "conservative",
        ]

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(ASSESS_PROMPT_TEMPLATE),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

        # 3. Define the conversation chain
        self.chain = chat_prompt | self.llm
        self.conversation = RunnableWithMessageHistory(
            self.chain,
            # Uses the get_by_session_id function defined in the example
            # above.
            get_by_session_id,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        self.user_input = None
        self.ask_user = True
        self.bot_response = {"destination": "question", "response": None}

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
            response = {"destination": "question", "response": response.content}
        self.bot_response = response
        self.bot_response["destination"] = self.bot_response["destination"].lower()

    def get_history(self):
        print(self.conversation.get_session_history(session_id=self.session_id))

    def __call__(self, user_input) -> Dict[str, Any]:
        if user_input is None:
            prompt = "Start the conversation with the first question. Remember to give at least 4 choices or examples separated into different line"
            self.predict(prompt)
            return self.bot_response

        # Ask question based on previous bot answer
        if self.bot_response["destination"] == "question":
            prompt = f"User: {user_input}.\n Instruction: Response a bit to last question, then ask next question. Remember to give at least 4 choices or examples separated into different line."
            self.predict(prompt)
        elif self.bot_response["destination"] == "explanation":
            prompt = f"User: {user_input}.\n Instruction: Response to what user asked. Follow by asking previous question again."
            self.predict(prompt)
        elif self.bot_response["destination"] == "unrelated":
            prompt = f"User: {user_input}.\n Instruction: Response gently to what they said, then resume asking last unanswered question."
        elif (
            self.bot_response["destination"] == "result"
        ):  # This only reach if result content is not statisfied.
            prompt = f"Please rephrase result again. Give exactly one of the following answers:  Aggressive, Moderate-aggressive, Moderate, Moderate-conservative, Conservative. No extra response!! "
            self.predict(prompt)
        else:
            prompt = f"Please rephrase your previous result again and return destination value with exactly one of followings: 'Question', 'Explanation', 'Result'"
            self.predict(prompt)

        return self.bot_response


class GeneralBot:
    def __init__(self, llm, session_id) -> None:
        self.llm = llm
        self.session_id = session_id
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(MOFU_PROMPT_TEMPLATE),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

        self.chain = chat_prompt | self.llm
        self.conversation = RunnableWithMessageHistory(
            self.chain,
            # Uses the get_by_session_id function defined in the example
            # above.
            get_by_session_id,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def get_history(self):
        print(self.conversation.get_session_history(session_id=self.session_id))

    def __call__(self, user_input):
        router_output = self.conversation.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": self.session_id}},
        )
        try:
            response = json.loads(router_output.content)
        except json.JSONDecodeError:
            # Placeholder
            print(
                f"WARNING: Invalid decoding to dict from output text: {router_output.content}"
            )
            response = {
                "destination": "general_chat",
                "response": router_output.content,
            }
        return response


class MofuChatBot:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(temperature=0.7)
        self.session_id = 0
        self.reset()

    def reset(self):
        for k in store:
            store[k].clear()
        self.conversations = {
            "general_chat": GeneralBot(self.llm, f"general-{self.session_id}"),
            "risk_assessment": RiskAssessBot(
                self.llm, f"risk-assessment-{self.session_id}"
            ),
            "asset_allocation": None,
        }
        self.mode = "general_chat"
        self.current_conversation = self.conversations["general_chat"]

    def set_status(self, mode: str):
        """
        Mode: ["general_chat", "risk_assessment", "asset_allocation"]
        """
        if mode in self.conversations:
            self.current_conversation = self.conversations[mode]
            self.mode = mode
            print(f"MofuBot switch to {self.mode}")
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected: {self.conversations}")

    def chat(self, user_input: str):
        output = self.current_conversation(user_input)
        if self.mode == "general_chat":
            if output["destination"].lower() == "risk_assessment":
                self.set_status("risk_assessment")
                output_init = self.current_conversation(None)

                return output["response"] + "\n" + output_init["response"]

            return output["response"]
        elif self.mode == "risk_assessment":
            if output["destination"] == "result":
                self.set_status("general_chat")

            return output["response"]
        else:
            raise NotImplementedError()
