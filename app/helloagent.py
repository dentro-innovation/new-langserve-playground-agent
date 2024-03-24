from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from typing import List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.runnables import RunnableLambda

load_dotenv(find_dotenv())

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant who makes fun of Swift developers."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

gpt4 = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)

agent_tools = [DuckDuckGoSearchRun()]
langchain_agent = create_openai_tools_agent(gpt4, agent_tools, prompt)
agent_executor = AgentExecutor(agent=langchain_agent, tools=agent_tools, return_intermediate_steps=True)

class Input(BaseModel):
    chat_history: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(default_factory=list)
    input: str

def parse_agent_output(agent_output):
    return agent_output["output"]

chain = (agent_executor | RunnableLambda(parse_agent_output)).with_types(input_type=Input, output_type=str)