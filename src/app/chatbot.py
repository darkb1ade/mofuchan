import json
import os
from typing import List

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


def create_investment_advisor():
    # 1. Set up the OpenAI language model
    llm = ChatOpenAI(temperature=0.7)

    # 2. Create a custom prompt template
    system_template = """You are an expert in finanicial investment advisor. \
            Your task is to ask a series of questions of at least 10 quesitons to determine the appropriate investment strategy for the user. \
            Question comes with at least four choices to answer. \
            Ask one question at a time and wait for the user's response before proceeding to the next question. \
            If user ask for explanation, please clarify their request. \
            If user response is not related to question, do not answer and ask previous question until related answer is received. \
            After gathering sufficient information, provide an investment strategy recommendation which MUST BE EXACTLY as one of following: Aggressive, Moderate-aggressive, Moderate, Moderate-conservative, Conservative.\

            IMPORTANT: Always format your response as a JSON string with the following structure:
            {{{{
                "response_status": "One of following: ["Question", "Explanation", "Result"]",
                "response": "Response to the user input".
            }}}}

            response_status explanation: REMEMBER: it has to be one of followings
            - Question: When user response to previous question and you want to ask next question
            - Explanation: When user want to clarify previous question more.
            - Result: When user response enough question to decide their investment strategy which MUST BE EXACTLY as one of following: Aggressive, Moderate-aggressive, Moderate, Moderate-conservative, Conservative. No extra response!!
            """

    human_template = "{input}"

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(human_template),
        ]
    )

    # 3. Define the conversation chain
    chain = chat_prompt | llm
    conversation = RunnableWithMessageHistory(
        chain,
        # Uses the get_by_session_id function defined in the example
        # above.
        get_by_session_id,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # 4. Implement the main function to run the conversation
    def run_conversation():
        print(
            "AI: Hello! I'm here to help determine an appropriate investment strategy for you. Let's begin with a few questions."
        )

        cnt = 0

        user_input = None
        response_status = "question"
        ask_user = True
        response = None

        expected_output = [
            "aggressive",
            "moderate-aggressive",
            "moderate",
            "moderate-conservative",
            "conservative",
        ]
        while True:
            if user_input is None:
                prompt = "Ask first question."
            else:
                if response_status.lower() == "question":
                    prompt = f"User: {user_input}.\n Ask next question, or provide result if you have enough information."
                    ask_user = True
                elif response_status.lower() == "explanation":
                    prompt = f"User: {user_input}.\n Response to what user asked. Follow by asking previous question again."
                    ask_user = True
                elif (
                    response_status.lower() == "result"
                ):  # This only reach if result content is not statisfied.
                    prompt = f"Please rephrase result again. Give exactly one of the following answers:  Aggressive, Moderate-aggressive, Moderate, Moderate-conservative, Conservative. No extra response!! "
                    ask_user = False
                else:
                    prompt = f"Please rephrase your previous result again and return response_status value with exactly one of followings: 'Question', 'Explanation', 'Result'"
                    ask_user = False

            response = conversation.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": "0"}},
            )

            try:
                response = json.loads(response.content)
            except json.JSONDecodeError:
                # Placeholder
                print(
                    f"WARNING: Invalid decoding to dict from output text: {response.content}"
                )
                response = {"response_status": "question", "response": response.content}

            response_status = response["response_status"]
            answer = response["response"]

            print(f"AI response status: {response_status}")
            print(f"AI said: {answer}")

            if response_status.lower() == "result":
                if answer.lower() in expected_output:
                    print(f"Your investment strategy level is: {answer}")
                    break
                else:
                    print(f"!!!!!!!Found result not match format: {answer}!!!!!!!!!!")

            # print(
            #     "History: \n ----- \n",
            #     conversation.get_session_history(session_id="0"),
            #     "----\nEND\n",
            # )

            if ask_user:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("AI: Thank you for your time. Goodbye!")
                    break

    return run_conversation


# Usage
investment_advisor = create_investment_advisor()
investment_advisor()
