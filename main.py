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


tools = []
tools.append(TavilySearch())


# llm = ChatOpenAI(model="gpt-5", temperature=0)
llm = ChatOllama(model="qwen3:1.7b", temperature=0)
agent = create_agent(model=llm, tools=tools)


def main():
    # response = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="عايز ايات من القران الكريم بتتكلم عن الميراث وعايز تفسيرها "
                )
            ]
        }
    )
    print(response)


if __name__ == "__main__":
    main()
