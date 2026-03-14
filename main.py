from typing import List 
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()
# from tavily import TavilyClient
# tavily_client = TavilyClient()
# @tool
# def search(query: str) -> str:
#    """Tool that searches over internet"""
#    print(f"Searching for {query}")
#    return tavily_client.search(query=query)
# tools = [search]


class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the response from the agent"""
    sources: List[Source] = Field(description="The sources used by the agent",default_factory=list)
    answer: str = Field(description="The answer to the question")


tools = []
tools.append(TavilySearch())

llm = ChatOllama(model="qwen3:1.7b", temperature=0, format=AgentResponse.model_json_schema())
llm = ChatOpenAI(model="gpt-5.2", temperature=0)

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    # response = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="List me 10 opening job vacancies in Egypt Related to Backend Development"
                )
            ]
        }
    )

if __name__ == "__main__":
    main()
