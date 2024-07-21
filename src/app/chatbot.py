import json
import os
from typing import Any, Dict, List
from nltk.tokenize import RegexpTokenizer

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
HISTORY_STORE = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in HISTORY_STORE:
        HISTORY_STORE[session_id] = InMemoryHistory()
    return HISTORY_STORE[session_id]


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
        if self.bot_response["destination"] == "result":
            final_response = self.post_process_result(self.bot_response["response"])
            if final_response is None:
                # Give one more chance
                prompt = f"Instruction: Return user risk level again. Give exactly one of the following answers:  Aggressive, Moderate-aggressive, Moderate, Moderate-conservative, Conservative. No extra response!!"
                self.predict(prompt)
                final_response = self.post_process_result(self.bot_response["response"])
                if final_response is None:
                    final_response = "Can you repeat the last answer again?"
            self.bot_response["response"] = final_response

    def get_history(self):
        print(self.conversation.get_session_history(session_id=self.session_id))

    def post_process_result(self, response_text):
        tokenizer = RegexpTokenizer(r"\w+")
        norm_candidates = []
        for risk_level_option in self.risk_level_options:
            norm_candidates.extend(tokenizer.tokenize(risk_level_option.lower()))
        norm_candidates = list(set(norm_candidates))
        response_text = tokenizer.tokenize(response_text.lower())
        outs = list(set(response_text).intersection(norm_candidates))

        if len(outs) == 1:
            return outs[0].capitalize()
        if len(outs) == 2:
            return ("-").join(["Moderate"] + [out for out in outs if out != "moderate"])
        else:  # wrongly detected risk-lvel
            return None

        # response_text = response_text.lower()
        # risk_level_options = sorted(self.risk_level_options, key=len)
        # for risk_level in risk_level_options:
        #     if risk_level.lower() in response_text:
        #         return risk_level
        # return None

    def __call__(self, user_input) -> Dict[str, Any]:
        if user_input is None:
            prompt = "Please give me the first question. Only for this time, no extra response in the beginning. Remember to give at least 4 choices or examples separated into different line"
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
            prompt = f"User: {user_input}.\n Inasdastruction: Response politely to what they said, then resume asking last unanswered question."
            self.predict(prompt)
        elif (
            self.bot_response["destination"] == "result"
        ):  # This only reach if result content is not statisfied.
            prompt = f"User: {user_input}.\n Instruction: Return user risk level based on their answer. Give exactly one of the following answers:  Aggressive, Moderate-aggressive, Moderate, Moderate-conservative, Conservative. No extra response!! "
            self.predict(prompt)
        else:
            prompt = f"Instruction: Please rephrase your previous result again and return destination value with exactly one of followings: 'Question', 'Explanation', 'Result'"
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

        print(f"Debug: General Chat : {response}")

        # Mofu-chan want to move to asset allocation model but not finalize risk level.
        if (
            response["destination"] == "portfolio"
            and response["response"] not in self.risk_level_options
        ):
            new_response = self(
                f"Instruction: Specify user risk level with EXACTLY ONE VALUE from of followings: {self.risk_level_options}."
            )
            # If you really don't have enough info, return destination: general_chat and response with question to ask user for their investment risk level instead DONT ASSUME ANSWER."
            # )
            print(f"Unexpected profile: {response}")
            if new_response["response"] not in self.risk_level_options:
                response = {
                    "destination": "general_chat",
                    "response": new_response["response"],
                }
            else:
                response["response"] = new_response["response"]

        return response


class ProfileAllocateBot:
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

        self.asset_name = ["income_assets", "commodity", "currency", "equity"]

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
        print(f"Default profile: {self.profile_value}")
        return self(
            "Instruction: Send message recommending given profile and ask for adjustment."
        )

    def predict(self, prompt: str):
        response = self.conversation.invoke(
            {
                "input": prompt,
                "risk_assess_level": self.risk_assess_level,
                "portfolio": self.profile_value,
            },
            config={"configurable": {"session_id": self.session_id}},
        )
        try:
            response = json.loads(response.content)
        except json.JSONDecodeError:
            # Placeholder
            print(
                f"WARNING: Invalid decoding to dict from output text: {response.content}"
            )
            response = {
                "destination": "discussion",
                "response": response.content,
            }
        print(f"Debug: Profile allocation : {response}")
        return response

    def convert_output_to_dict(self, output_string: str):
        output_value = output_string.replace("\n", "").strip().split("-")
        if len(output_value) != 5:
            print(f"Error: Invalid format for portfolio output: {output_string}")
            return None

        output_dict = {}
        for val in output_value[1:]:
            if val.count(":") != 1:
                print(f"Error: Invalid format for portfolio output: {output_string}")
                return None
            asset_key, asset_value = val.split(":")[0], val.split(":")[1]
            try:
                output_dict[asset_key.strip()] = (
                    float(asset_value.replace("%", "").strip()) / 100
                )
            except ValueError:
                print(f"Error: Invalid format for portfolio output: {output_string}")
                return None
        return output_dict

    def __call__(self, user_input):
        response = self.predict(user_input)

        # Handle error
        if response["destination"] == "result":
            dict_output = self.convert_output_to_dict(response["response"])
            if dict_output is None:
                confirm_response = self.predict(
                    f"Instruction: Confirm the final profile value exactly in the given format: {self.profile_value}"
                )
                dict_output = self.convert_output_to_dict(confirm_response["response"])

            response["response"] = dict_output

        return response


class MofuChatBot:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
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
        for k in HISTORY_STORE:
            HISTORY_STORE[k].clear()
        self.bots = {
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
            "profile_allocation": ProfileAllocateBot(
                self.llm, f"asset-allo-{self.session_id}", self.default_profile
            ),
        }
        self.mode = "general_chat"
        self.current_bot = self.bots["general_chat"]

    def set_status(self, mode: str):
        """
        Mode: ["general_chat", "risk_assessment", "profile_allocation"]
        """
        if mode in self.bots:
            self.current_bot = self.bots[mode]
            self.mode = mode
            print(f"MofuBot switch to {self.mode}")
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Expected: {(',').join(list(self.converstions.keys()))}"
            )

    def get_history(self):
        self.current_bot.get_history()

    def chat(self, user_input: str):
        output = self.current_bot(user_input)
        if self.mode == "general_chat":
            if output["destination"].lower() == "risk_assessment":
                self.set_status("risk_assessment")
                output_init = self.current_bot(None)

                return (
                    "Let's get started with risk assessment then!"
                    + "\n"
                    + output_init["response"]
                )
            elif output["destination"].lower() == "portfolio":
                self.set_status("profile_allocation")
                output_init = self.current_bot.init_conversation(output["response"])
                return output_init["response"]
            else:
                return output["response"]
        elif self.mode == "risk_assessment":
            if output["destination"] == "result":
                self.set_status("profile_allocation")
                output_init = self.current_bot.init_conversation(output["response"])
                return output_init["response"]
            else:
                return output["response"]
        elif self.mode == "profile_allocation":
            if output["destination"] == "result":
                self.set_status("general_chat")
                return output["response"]
            else:
                return output["response"]
        else:
            raise NotImplementedError()
