from dotenv import load_dotenv
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langchain_tavily import TavilySearch

load_dotenv()

# @tool
# def get_weather(city: str) -> str:
#     """Get the current weather for a specific city.
#        Always use this tool when asked about weather.

#     Args:
#         city: The name of the city to get weather for.

#     Returns:
#         Current weather conditions for that city.
#     """
#     print(f"Getting weather for {city}")
#     return tavily.search(query=city)


llm = ChatAnthropic(model="claude-opus-4-5")
tools = [TavilySearch()]
agent = create_react_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-learning!")
    langfuse_handler = CallbackHandler()
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in NYC?")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(result["messages"][-1].content)
    get_client().flush()


if __name__ == "__main__":
    main()