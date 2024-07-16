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
    def __init__(self, llm, session_id, risk_level_options) -> None:
        self.llm = llm
        self.session_id = session_id
        self.risk_level_options = risk_level_options

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
    def __init__(self, llm, session_id, risk_level_options) -> None:
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
        self.risk_level_options = risk_level_options

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

        # Mofu-chan want to move to asset allocation model but not finalize risk level.
        if (
            response["destination"] == "asset_allocation"
            and response["response"] not in self.risk_level_options
        ):
            new_response = self(
                f"Instruction: please specify user risk level with one exactly one of followings: {self.risk_level_options}. \
                                If you don't have enough info, return destination: general_chat and response with question to ask user for their investment risk level instead DONT ASSUME ANSWER."
            )
            print(f"Unexpected asset_allocation: {new_response}")
            if new_response["response"] not in self.risk_level_options:
                response = {
                    "destination": "general_chat",
                    "response": new_response["response"],
                }
            else:
                response["response"] = new_response["response"]

        return response


class AssetAllocateBot:
    def __init__(self, llm, session_id, default_allocation):
        self.llm = llm
        self.session_id = session_id
        self.default_allocation = default_allocation
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(PROFILE_PROMPT_TEMPLATE),
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

        self.asset_name = ["income_assets", "commodities", "currencies", "equities"]

        self.risk_assess_level = None
        self.profile_value = None

    def get_history(self):
        print(self.conversation.get_session_history(session_id=self.session_id))

    def init_conversation(self, risk_assess_level):
        self.risk_assess_level = risk_assess_level
        self.profile_value = "\n".join(
            [
                f"- {name} : {value*100}%"
                for name, value in zip(
                    self.asset_name, self.default_allocation[risk_assess_level]
                )
            ]
        )
        print(f"Asset alllocation: {self.profile_value}")
        return self(
            "Instruction: Send message asking if given asset allocation is acceptable and open for adjustment."
        )

    def __call__(self, user_input):
        router_output = self.conversation.invoke(
            {
                "input": user_input,
                "risk_assess_level": self.risk_assess_level,
                "profile": self.profile_value,
            },
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
                "destination": "discussion",
                "response": router_output.content,
            }

        # Handle error
        if response["destination"] == "result":
            confirm_response = self.conversation.invoke(
                {
                    "input": f"Instruction: Confirm the final profile value exactly in the given format: {self.risk_assess_level}",
                    "risk_assess_level": self.risk_assess_level,
                    "profile": self.profile_value,
                }
            )
            response["result"] = confirm_response["result"]

        return response


class MofuChatBot:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(temperature=0.7)
        self.session_id = 0
        self.default_profile = {
            "conservative": [0.7, 0.05, 0.05, 0.2],
            "moderate-conservative": [0.6, 0.05, 0.05, 0.3],
            "moderate": [0.5, 0.1, 0.05, 0.35],
            "moderate-aggressive": [0.3, 0.1, 0.05, 0.55],
            "aggressive": [0.1, 0.1, 0.05, 0.75],
        }
        self.reset()

    def reset(self):
        for k in store:
            store[k].clear()
        self.conversations = {
            "general_chat": GeneralBot(
                self.llm,
                f"general-{self.session_id}",
                list(self.default_profile.keys()),
            ),
            "risk_assessment": RiskAssessBot(
                self.llm,
                f"risk-assessment-{self.session_id}",
                list(self.default_profile.keys()),
            ),
            "asset_allocation": AssetAllocateBot(
                self.llm, f"asset-allo-{self.session_id}", self.default_profile
            ),
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
            elif output["destination"].lower() == "asset_allocation":
                self.set_status("asset_allocation")
                output_init = self.current_conversation.init_conversation(
                    output["response"]
                )
                return output["response"] + "\n" + output_init["response"]
            else:
                return output["response"]
        elif self.mode == "risk_assessment":
            if output["destination"] == "result":
                self.set_status("asset_allocation")
                output_init = self.current_conversation.init_conversation(
                    output["response"]
                )
                return output["response"] + "\n" + output_init["response"]
            else:
                return output["response"]
        elif self.mode == "asset_allocation":
            if output["destination"] == "result":
                self.set_status("general_chat")
                return f"Final output is: {output['response']}"
            else:
                return output["response"]
        else:
            raise NotImplementedError()
